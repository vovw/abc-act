import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


from abc_minimal.episode_io import discover_episodes, load_episode
from abc_minimal.dataloader import decode_frame
from abc_minimal.preprocess import resize_pad_normalize, augment_and_normalize





class ACTDataset(Dataset):
    """All timesteps across selected episodes, with episode-local action chunks."""

    def __init__(self, episode_dirs, chunk_size=20, norm_stats=None, augment=False):
        if isinstance(episode_dirs, (str, Path)):
            episode_dirs = [episode_dirs]
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.augment = augment
        self.camera_keys = ("top", "left", "right")
        self.episodes = []
        for path in episode_dirs:
            path = Path(path)
            metadata, states, actions = load_episode(path)
            self.episodes.append((path, metadata, states, actions))
        if not self.episodes:
            raise ValueError("No episodes selected")
        self.ends = np.cumsum([len(ep[2]) for ep in self.episodes])

        # Training computes these once; validation reuses the exact same stats.
        if norm_stats is None:
            norm_stats = {}
            for name, column in (("state", 2), ("action", 3)):
                count, mean, m2 = 0, np.zeros(14), np.zeros(14)
                for episode in self.episodes:
                    values = episode[column]
                    n = len(values)
                    delta = values.mean(axis=0) - mean
                    total = count + n
                    m2 += ((values - values.mean(axis=0)) ** 2).sum(axis=0)
                    m2 += delta ** 2 * count * n / total
                    mean += delta * n / total
                    count = total
                norm_stats[name + "_mean"] = mean
                norm_stats[name + "_std"] = np.sqrt(m2 / count).clip(min=1e-6)
        # Keep workers small; compute original float64 statistics before casting.
        self.episodes = [(p, m, st.astype(np.float32), ac.astype(np.float32))
                         for p, m, st, ac in self.episodes]
        self.norm_stats = {key: np.asarray(value) for key, value in norm_stats.items()}
        for key, value in self.norm_stats.items():
            setattr(self, key, value)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        start = int(self.ends[episode_index - 1]) if episode_index else 0
        t = index - start
        path, metadata, states, actions = self.episodes[episode_index]
        valid_count = min(self.chunk_size, len(states) - t)
        state = (states[t] - self.state_mean) / self.state_std
        chunk = np.zeros((self.chunk_size, 14), dtype=np.float32)
        chunk[:valid_count] = (
            actions[t:t + valid_count] - self.action_mean
        ) / self.action_std
        images = decode_frame(
            path, t, len(states), metadata["cameras"], self.camera_keys
        )
        # getattr also supports datasets pickled before this option was added.
        if getattr(self, "augment", False):
            images = augment_and_normalize(images, train=True)
        else:
            images = {cam: resize_pad_normalize(images[cam]) for cam in self.camera_keys}
        return {
            "images": torch.stack([images[cam] for cam in self.camera_keys]),
            "state": torch.as_tensor(state, dtype=torch.float32),
            "actions": torch.from_numpy(chunk),
            "is_pad": torch.from_numpy(np.arange(self.chunk_size) >= valid_count),
        }


class RenderedACTDataset(Dataset):
    """Memory-mapped live-rendered observations with demonstration action targets."""

    def __init__(self, root, split, norm_stats, augment=False):
        import hashlib
        root = Path(root)
        manifest_bytes = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.data_signature = hashlib.sha256(manifest_bytes).hexdigest()
        if set(manifest["splits"]) != {"train", "val"}:
            raise ValueError("Rendered dataset generation has not completed both splits")
        self.directory = root / split
        self.chunk_size = manifest["chunk_size"]
        self.camera_keys = tuple(manifest["camera_keys"])
        if self.camera_keys != ("top", "left", "right"):
            raise ValueError("Rendered camera order must be top, left, right")
        self.length = manifest["splits"][split]["samples"]
        self.augment = augment
        self.norm_stats = {k: np.asarray(v) for k, v in norm_stats.items()}
        self.episodes = []
        for name in manifest["splits"][split]["episodes"]:
            path = Path(name)
            metadata = json.loads((path / "episode_metadata.json").read_text())
            metadata["render_backend"] = manifest["camera_backend"]
            self.episodes.append((path, metadata, None, None))
        self._arrays = None
        for name, shape in (("images", (self.length, 3, 3, 168, 224)),
                            ("states", (self.length, 14)),
                            ("actions", (self.length, self.chunk_size, 14)),
                            ("is_pad", (self.length, self.chunk_size))):
            if np.load(self.directory / f"{name}.npy", mmap_mode="r").shape != shape:
                raise ValueError(f"Unexpected rendered {name} shape")

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        # Spawned workers reopen the files instead of pickling gigabytes.
        state["_arrays"] = None
        return state

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._arrays is None:
            self._arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r")
                            for name in ("images", "states", "actions", "is_pad")}
        raw = self._arrays
        images = {cam: torch.from_numpy(raw["images"][index, i].copy()).float() / 255
                  for i, cam in enumerate(self.camera_keys)}
        if self.augment:
            images = augment_and_normalize(images, train=True)
        else:
            images = {cam: resize_pad_normalize(value) for cam, value in images.items()}
        state = (raw["states"][index] - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
        actions = (raw["actions"][index] - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        is_pad = raw["is_pad"][index].copy()
        actions[is_pad] = 0
        return dict(images=torch.stack([images[cam] for cam in self.camera_keys]),
                    state=torch.as_tensor(state, dtype=torch.float32),
                    actions=torch.as_tensor(actions, dtype=torch.float32),
                    is_pad=torch.from_numpy(is_pad))


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)


        # remmove avg pooling and classfication keep spacial features.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(512, hidden_dim, kernel_size=1)

        self.camera_embedding = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.row_embedding = nn.Parameter(torch.randn(7, hidden_dim) * 0.02)
        self.col_embedding = nn.Parameter(torch.randn(7, hidden_dim) * 0.02)

    def forward(self, images):

        batch_size, num_cameras, channels, height, width = images.shape

        images = images.reshape(
                batch_size * num_cameras, channels, height, width
                )

        features = self.backbone(images)
        features = self.projection(features)

        _, hidden_dim, grid_h, grid_w = features.shape
        assert num_cameras == 3 and (grid_h, grid_w) == (7,7)

        features = features.reshape(
                batch_size, num_cameras, hidden_dim, grid_h, grid_w)

        features = features.permute(0, 1, 3, 4, 2)


        positions = (
                self.camera_embedding[:, None, None, :]
                + self.row_embedding[None,:, None, :]
                + self.col_embedding[None, None, :, :]
            )

        tokens = features + positions.unsqueeze(0)


        return tokens.reshape(
                batch_size,
                num_cameras * grid_h * grid_w,
                hidden_dim

                )


class ObservationEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim)
        self.state_projection = nn.Linear(14, hidden_dim)

    def forward(self, images, state):
        image_tokens = self.image_encoder(images)

        state_tokens = self.state_projection(state).unsqueeze(1)

        return torch.cat([state_tokens, image_tokens], dim=1)



class LatentEncoder(nn.Module):
    def __init__(self, chunk_size=20, hidden_dim=256, latent_dim=32):
        super().__init__()

        self.state_projection = nn.Linear(14, hidden_dim)
        self.action_projection = nn.Linear(14, hidden_dim)

        self.summary_token = nn.Parameter(torch.randn(1,1,hidden_dim) * 0.02)

        # one summary token + one state token + chunk_size action tokens
        self.position_embedding = nn.Parameter(
                    torch.randn(1, chunk_size + 2, hidden_dim) * 0.02
                )
        layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                )

        self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=2,
                enable_nested_tensor=False,
                )


        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)




    def forward(self, state, action, is_pad):
        batch_size, chunk_size, _ = action.shape

        state_token = self.state_projection(state).unsqueeze(1)
        #[b,1,256]
        action_token = self.action_projection(action)
        #[b,20,256]

        summary_token = self.summary_token.expand(batch_size, -1, -1)

        tokens = torch.cat(
                [summary_token, state_token, action_token], dim=1
                )

        tokens = tokens + self.position_embedding[:, :chunk_size +2]

        prefix_mask = torch.zeros(
                batch_size, 2, dtype=torch.bool, device=state.device)

        padding_mask = torch.cat([prefix_mask, is_pad], dim=1)

        encoded = self.transformer(
                tokens,
                src_key_padding_mask = padding_mask,
            )

        summary = encoded[:,0]
        mu = self.mu_head(summary)
        logvar = self.logvar_head(summary)

        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)


        z = mu+std*noise

        return z, mu, logvar


class ActionDecoder(nn.Module):
    def __init__(self, chunk_size=20, hidden_dim=256, latent_dim=32):
        super().__init__()

        self.latent_projection = nn.Linear(latent_dim, hidden_dim)


        self.action_queries = nn.Parameter(
                    torch.randn(1, chunk_size, hidden_dim) * 0.02
                )

        layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout = 0.1,
                batch_first = True,

                )
        self.transformer = nn.TransformerDecoder(
                    layer,
                    num_layers=2,
                )

        self.action_head = nn.Linear(hidden_dim, 14)

    def forward(self, observation_token, z):
        batch_size = observation_token.shape[0]

        latent_token = self.latent_projection(z).unsqueeze(1)

        memory = torch.cat(
                    [latent_token, observation_token],
                    dim=1
                )

        queries = self.action_queries.expand(batch_size, -1, -1)

        decoded = self.transformer(
                tgt = queries,
                memory = memory
                )

        predicted_actions = self.action_head(decoded)



        return predicted_actions



def compute_loss(predicted_actions, actions, is_pad, mu, logvar, kl_weight=10.0):
    valid = (~is_pad).unsqueeze(-1).to(predicted_actions.dtype)

    error = (predicted_actions - actions).abs()

    count = valid.sum() * actions.shape[-1]
    l1 = (error * valid).sum() / count.clamp_min(1)

    kl = -0.5 * (
            1 + logvar - mu.square() - logvar.exp()
            ).sum(dim=-1).mean()

    loss = l1 + kl_weight * kl

    return loss, l1, kl



class ACTPolicy(nn.Module):
    def __init__(self, chunk_size=20, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.observation_encoder = ObservationEncoder(hidden_dim)
        self.latent_encoder = LatentEncoder(chunk_size, hidden_dim, latent_dim)
        self.action_decoder = ActionDecoder(chunk_size, hidden_dim, latent_dim)
        self.freeze_batchnorm = False

    def configure_batchnorm(self, freeze):
        """Optionally retain fixed backbone statistics and affine parameters."""
        self.freeze_batchnorm = freeze
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                for parameter in module.parameters():
                    parameter.requires_grad_(not freeze)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_batchnorm", False):
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images, state, actions=None, is_pad=None):
        observation_tokens = self.observation_encoder(images, state)

        if actions is not None:
            # training: infer z from the demo
            if is_pad is None:
                raise ValueError("is_pad is required with actions")

            z, mu, logvar = self.latent_encoder(state, actions, is_pad)
        else:
            # infernece: no demo actions avail
            z = state.new_zeros(state.shape[0], self.latent_dim)
            mu, logvar = None, None

        prediction = self.action_decoder(observation_tokens, z)

        return prediction, mu, logvar






class CursorBatchSampler:
    """Deterministic shuffle with a sample cursor, independent of worker prefetch."""

    def __init__(self, size, batch_size, seed, samples_seen=0, max_batches=None):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.samples_seen, self.max_batches = samples_seen, max_batches

    def __iter__(self):
        epoch, offset = divmod(self.samples_seen, self.size)
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch))
        batches = 0
        while offset < self.size and (self.max_batches is None or batches < self.max_batches):
            end = min(offset + self.batch_size, self.size)
            yield order[offset:end].tolist()
            offset = end
            batches += 1

    def __len__(self):
        left = self.size - self.samples_seen % self.size
        count = (left + self.batch_size - 1) // self.batch_size
        return min(count, self.max_batches) if self.max_batches is not None else count


def select_episodes(root, task):
    return [path for path in discover_episodes(root)
            if json.loads((path / "episode_metadata.json").read_text()).get("task_name") == task]


@torch.no_grad()
def evaluate(policy, loader, device):
    """Masked L1 with z=0, without conditioning on target actions."""
    policy.eval()
    error_sum, count = 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        predictions, _, _ = policy(batch["images"], batch["state"])
        valid = (~batch["is_pad"]).unsqueeze(-1)
        error_sum += ((predictions - batch["actions"]).abs() * valid).sum().item()
        count += valid.sum().item() * 14
    policy.train()
    return error_sum / count


class ACTSimPolicy:
    """Convert simulator observations to checkpoint inputs and restore action units."""

    def __init__(self, checkpoint_path, device):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = ACTPolicy(**checkpoint["model_config"]).to(self.device).eval()
        self.model.load_state_dict(checkpoint["model"])
        self.camera_keys = tuple(checkpoint["camera_keys"])
        if self.camera_keys != ("top", "left", "right"):
            raise ValueError("This image encoder requires top, left, right camera order")
        self.chunk_size = checkpoint["model_config"]["chunk_size"]
        self.stats = {k: torch.tensor(v, dtype=torch.float32, device=self.device)
                      for k, v in checkpoint["norm_stats"].items()}
        self.data_fps = float(checkpoint["data_fps"])
        self.task = checkpoint["config"]["task"]
        self.step = checkpoint["step"]
        self.camera_height = checkpoint.get("camera_height", 224)
        self.camera_width = checkpoint.get("camera_width", 224)
        # The sim_224 bottle release labels frames "mjwarp", but pixel checks
        # against its saved scenes match classic MuJoCo much more closely with
        # the installed renderer versions. Preserve the metadata separately.
        self.data_camera_backend = checkpoint.get("camera_backend", "mujoco")
        self.camera_backend = checkpoint.get(
            "inference_camera_backend",
            "mujoco" if self.task == "sim_put_the_plastic_bottles_in_the_bin"
            else self.data_camera_backend,
        )

    @torch.inference_mode()
    def infer(self, obs):
        from abc_minimal.preprocess import resize_pad_normalize

        images = torch.stack([
            resize_pad_normalize(obs["images"][cam], target_h=224, target_w=224)
            for cam in self.camera_keys
        ]).unsqueeze(0).to(self.device)
        state = torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device)
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        prediction, _, _ = self.model(images, state.unsqueeze(0))
        actions = prediction[0] * self.stats["action_std"] + self.stats["action_mean"]
        if not torch.isfinite(actions).all():
            raise RuntimeError("Policy produced non-finite actions")
        return actions.cpu().numpy()


class TemporalEnsembler:
    """Average overlapping predictions, oldest first, as in the ACT reference."""

    def __init__(self, coefficient=0.01):
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("Temporal coefficient must be finite and nonnegative")
        self.coefficient = coefficient
        self.pending = []

    def reset(self):
        self.pending.clear()

    def update(self, chunk):
        self.pending.append(np.asarray(chunk).copy())
        weights = np.exp(-self.coefficient * np.arange(len(self.pending)))
        action = np.average(np.stack([p[0] for p in self.pending]), axis=0, weights=weights)
        self.pending = [p[1:] for p in self.pending if len(p) > 1]
        return action.astype(np.float32)


def run_sim_viewer(args):
    """ACT-controlled MuJoCo rollout, optionally streamed to a local Viser UI."""
    import os
    import time
    import threading

    os.environ.setdefault("MUJOCO_GL", "egl")
    import abc_sim

    if args.checkpoint is None:
        raise ValueError("--sim requires --checkpoint /path/to/best.pt")
    torch.set_num_threads(4)
    policy = ACTSimPolicy(args.checkpoint, args.device)
    if args.execute_actions is None:
        args.execute_actions = policy.chunk_size
    if policy.task not in {
        "put_the_plastic_bottles_in_the_bin",
        "sim_put_the_plastic_bottles_in_the_bin",
    }:
        raise ValueError("This simulator adapter currently supports the put-bottles task only")
    if not 1 <= args.execute_actions <= policy.chunk_size:
        raise ValueError("--execute-actions must be between 1 and checkpoint chunk size")
    if args.sim_steps <= 0:
        raise ValueError("--sim-steps must be positive")
    if args.success_hold_steps < 0:
        raise ValueError("--success-hold-steps must be nonnegative")
    if (args.video or args.rollout_json) and not args.headless:
        raise ValueError("--video and --rollout-json require --headless")
    ensemble = None
    if args.temporal_ensemble is not None:
        ensemble = TemporalEnsembler(args.temporal_ensemble)
        args.execute_actions = 1
    scene_options = {}
    if args.sim_episode:
        from abc_minimal.episode_io import load_scene_xml, seed_initial_state
        metadata, initial_states, _ = load_episode(args.sim_episode)
        if metadata["task_name"] != policy.task:
            raise ValueError("Episode and checkpoint tasks differ")
        assets = Path(__file__).resolve().parents[3] / "abc/abc_sim/models/assets"
        scene_xml, _ = load_scene_xml(args.sim_episode, metadata, assets)
        if scene_xml is None:
            raise ValueError("The selected episode has no saved scene")
        scene_options.update(scene_xml_string=scene_xml, enable_task_randomizer=False)

    # Keep 17 physics substeps while matching the checkpoint's exact 30 Hz grid.
    env = abc_sim.make_env(
        task="put_plastic_bottles_in_bin", render_cameras=True,
        camera_backend=args.camera_backend or policy.camera_backend,
        camera_height=policy.camera_height, camera_width=policy.camera_width,
        physics_dt=1.0 / (policy.data_fps * 17), control_decimation=17,
        max_episode_steps=args.sim_steps, terminate_on_success=args.success_hold_steps == 0,
        **scene_options,
    )
    server = None
    video_process = None
    try:
        seed = args.seed
        obs, _ = env.reset(seed=seed, randomize=not bool(args.sim_episode))
        if args.sim_episode:
            seed_initial_state(env, args.sim_episode, metadata, initial_states)
            obs = env.get_obs()
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            video_process = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size",
                f"{policy.camera_width}x{policy.camera_height}",
                "-framerate", str(policy.data_fps), "-i", "-", "-an",
                "-vf", "scale=672:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-y", str(args.video),
            ], stdin=subprocess.PIPE)
        print(f"ACT checkpoint step {policy.step} | {args.device} | "
              f"{policy.data_fps:g} Hz | replan every {args.execute_actions} actions", flush=True)
        print(f"Camera renderer: {args.camera_backend or policy.camera_backend} "
              f"(dataset metadata: {policy.data_camera_backend})", flush=True)
        training_domain = "Simulation" if policy.task.startswith("sim_") else "Real-camera"
        print(f"{training_domain} checkpoint; rollout success is measured below.", flush=True)
        reset_requested = threading.Event()
        scene = None
        if not args.headless:
            import viser
            from mjviser import ViserMujocoScene
            server = viser.ViserServer(host="127.0.0.1", port=args.port, label="ACT rollout")

            @server.on_client_connect
            def set_view(client):
                client.camera.position = (-0.42, 0.0, 1.66)
                client.camera.look_at = (0.45, 0.0, 0.87)
                client.camera.up_direction = (0.0, 0.0, 1.0)

            server.gui.add_markdown(
                f"**ACT · put bottles in bin**\n\nCheckpoint step: {policy.step} · z = 0\n\n"
                f"Trained on {training_domain.lower()} data. Every action comes from ACT.\n\n"
                + (f"Starting scene: {args.sim_episode.name}" if args.sim_episode else "Starting scene: randomized simulator")
            )
            playing = server.gui.add_checkbox("Play", initial_value=False)
            reset_button = server.gui.add_button("Replay same initial scene" if args.sim_episode else "Reset with next seed")
            speed = server.gui.add_slider("Playback speed", min=0.25, max=2.0, step=0.25, initial_value=1.0)
            status = server.gui.add_text("Status", initial_value="Ready — press Play", disabled=True)
            camera_handles = {}
            with server.gui.add_folder("Policy camera inputs"):
                for cam in policy.camera_keys:
                    camera_handles[cam] = server.gui.add_image(
                        obs["images"][cam].transpose(1, 2, 0), label=cam, format="jpeg"
                    )

            @reset_button.on_click
            def request_reset(_):
                reset_requested.set()

            def build_scene():
                server.scene.reset()
                result = ViserMujocoScene(server, env.model, num_envs=1)
                result.camera_tracking_enabled = False
                server.scene.remove_by_name("/fixed_bodies/world/table_plane")
                result.update_from_mjdata(env.data)
                return result

            scene = build_scene()
            print(f"Viewer: http://127.0.0.1:{server.get_port()} — press Play", flush=True)

        steps, actions, action_index, done = 0, None, 0, False
        peak_reward, success, ever_success, success_streak = 0.0, False, False, 0
        rollout_started = time.monotonic()
        while True:
            if reset_requested.is_set():
                reset_requested.clear()
                if not args.sim_episode:
                    seed += 1
                previous_model = env.model
                env.forget_arm_state()
                obs, _ = env.reset(seed=seed, randomize=not bool(args.sim_episode))
                if args.sim_episode:
                    seed_initial_state(env, args.sim_episode, metadata, initial_states)
                    obs = env.get_obs()
                steps, actions, action_index, done = 0, None, 0, False
                peak_reward, success, ever_success, success_streak = 0.0, False, False, 0
                if ensemble is not None:
                    ensemble.reset()
                if scene is not None:
                    if env.model is not previous_model:
                        scene = build_scene()
                    else:
                        scene.update_from_mjdata(env.data)
                    for cam, handle in camera_handles.items():
                        handle.image = obs["images"][cam].transpose(1, 2, 0)
                    status.value = f"Seed {seed} — ready"
            if not args.headless and (not playing.value or done):
                time.sleep(0.02)
                continue
            started = time.perf_counter()
            if actions is None or action_index >= args.execute_actions:
                obs = env.get_obs()
                actions = policy.infer(obs)
                action_index = 0
                if scene is not None:
                    for cam, handle in camera_handles.items():
                        handle.image = obs["images"][cam].transpose(1, 2, 0)
            action = ensemble.update(actions) if ensemble is not None else actions[action_index]
            action_index += 1
            _, reward, terminated, truncated, info = env.step(action, render_obs=False)
            steps += 1
            done = terminated or truncated
            task_info = info.get("task_eval", {})
            instant_success = bool(info.get("task_success", False))
            ever_success |= instant_success
            success_streak = success_streak + 1 if instant_success else 0
            success = (success_streak >= args.success_hold_steps
                       if args.success_hold_steps else ever_success)
            if args.success_hold_steps and success:
                done = True
            peak_reward = max(peak_reward, float(reward))
            if video_process is not None:
                frame = env.get_obs()["images"]["top"].transpose(1, 2, 0)
                video_process.stdin.write(np.ascontiguousarray(frame).tobytes())
            if scene is not None:
                scene.update_from_mjdata(env.data)
                status.value = (f"Seed {seed} | {steps}/{args.sim_steps} | "
                                f"reward {float(reward):.3f} | success {success}")
            if steps == 1 or steps % 30 == 0 or done:
                print(f"Seed {seed} step {steps} reward={float(reward):.3f} success={success}", flush=True)
            if done:
                if args.headless:
                    break
                playing.value = False
                status.value += " — finished; reset to try another seed"
            if not args.headless:
                delay = 1.0 / (policy.data_fps * speed.value) - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
        if args.rollout_json:
            import hashlib
            report = dict(
                checkpoint=str(args.checkpoint.resolve()), checkpoint_step=policy.step,
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                seed=seed, episode=str(args.sim_episode) if args.sim_episode else None,
                steps=steps, max_steps=args.sim_steps, peak_reward=peak_reward,
                success=success, task_eval=task_info,
                ever_success=ever_success, success_hold_steps=args.success_hold_steps,
                consecutive_success_steps=success_streak,
                camera_backend=args.camera_backend or policy.camera_backend,
                execute_actions=args.execute_actions, temporal_ensemble=args.temporal_ensemble,
                video=str(args.video) if args.video else None,
                elapsed_seconds=time.monotonic() - rollout_started,
            )
            args.rollout_json.parent.mkdir(parents=True, exist_ok=True)
            args.rollout_json.write_text(json.dumps(
                report, indent=2, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)
            ))
    finally:
        try:
            if video_process is not None:
                video_process.stdin.close()
                if video_process.wait() != 0:
                    raise RuntimeError("ffmpeg could not write the rollout video")
        finally:
            env.close()
            if server is not None:
                server.stop()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Train ACT on multiple episodes of one ABC task.")
    parser.add_argument("--train-dir", type=Path, default=root / "abc/cache/train_real")
    parser.add_argument("--val-dir", type=Path, default=root / "abc/cache/val_real")
    parser.add_argument("--rendered-dir", type=Path,
                        help="Fine-tune on a prepared live-rendered buffer using checkpoint normalization")
    parser.add_argument("--task", default="put_the_plastic_bottles_in_the_bin")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int,
                        help="Make this many complete passes; overrides --steps")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Start from saved weights and matching normalization, with a new optimizer")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-recycle-every", type=int, default=1000,
                        help="Recreate video decoding workers after this many batches")
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Restore model/optimizer and saved sampler cursor when available")
    parser.add_argument("--lr", type=float,
                        help="Learning rate; defaults to 1e-4, or the checkpoint rate when resuming")
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None,
                        help="Apply ABC image augmentation; resume inherits the saved setting")
    parser.add_argument("--freeze-batchnorm", action=argparse.BooleanOptionalAction, default=None,
                        help="Freeze backbone normalization; resume inherits the saved setting")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path,
                        default=root / "checkpoints" / datetime.now().strftime("act-%Y%m%d-%H%M%S"))
    parser.add_argument("--sim", action="store_true", help="Run the ACT simulator viewer instead of training")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--execute-actions", type=int,
                        help="Actions to execute before replanning; defaults to the checkpoint chunk size")
    parser.add_argument("--sim-steps", type=int, default=2100)
    parser.add_argument("--success-hold-steps", type=int, default=0,
                        help="Continue ACT control until success holds for this many consecutive actions")
    parser.add_argument("--headless", action="store_true", help="Run one unpaced simulator episode without a browser")
    parser.add_argument("--sim-episode", type=Path,
                        help="Use a saved episode's scene and initial state; actions still come from ACT")
    parser.add_argument("--video", type=Path, help="Save the headless ACT rollout as an MP4")
    parser.add_argument("--rollout-json", type=Path, help="Save measured headless rollout results")
    parser.add_argument("--temporal-ensemble", type=float,
                        help="Average overlapping chunks with this coefficient; replan every action")
    parser.add_argument("--camera-backend", choices=("mujoco", "mjwarp"),
                        help="Override the simulator renderer saved in the checkpoint")
    args = parser.parse_args()
    if args.sim:
        run_sim_viewer(args)
        return
    if min(args.steps, args.batch_size, args.chunk_size, args.eval_every, args.eval_samples) <= 0:
        parser.error("steps, batch size, chunk size and evaluation settings must be positive")
    if args.worker_recycle_every <= 0:
        parser.error("worker-recycle-every must be positive")
    if args.init_checkpoint and args.resume_checkpoint:
        parser.error("Choose init-checkpoint or resume-checkpoint, not both")
    if args.workers < 0 or (args.lr is not None and args.lr <= 0) or args.kl_weight < 0:
        parser.error("workers and KL weight must be nonnegative; lr must be positive")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("epochs must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    initial_path = args.resume_checkpoint or args.init_checkpoint
    initial = torch.load(initial_path, map_location="cpu", weights_only=True) if initial_path else None
    if args.resume_checkpoint and initial.get("warm_start_only", False):
        parser.error("This converted checkpoint requires --init-checkpoint and a new optimizer")
    if args.rendered_dir is None and args.resume_checkpoint and initial["config"].get("rendered_dir"):
        args.rendered_dir = Path(initial["config"]["rendered_dir"])
    if args.rendered_dir:
        args.rendered_dir = args.rendered_dir.resolve()
        if initial is None:
            parser.error("Rendered fine-tuning requires init-checkpoint or resume-checkpoint")
        train_data = RenderedACTDataset(args.rendered_dir, "train", initial["norm_stats"], args.augment)
        val_data = RenderedACTDataset(args.rendered_dir, "val", initial["norm_stats"])
        train_paths = [ep[0] for ep in train_data.episodes]
        val_paths = [ep[0] for ep in val_data.episodes]
        if train_data.chunk_size != args.chunk_size:
            parser.error("Rendered action chunk size differs from --chunk-size")
        if any(ep[1]["task_name"] != args.task for data in (train_data, val_data) for ep in data.episodes):
            parser.error("Rendered buffer contains a different task")
    else:
        train_paths = select_episodes(args.train_dir, args.task)
        val_paths = select_episodes(args.val_dir, args.task)
    if not train_paths or not val_paths:
        parser.error("The selected task must have both training and validation episodes")
    if {p.name for p in train_paths} & {p.name for p in val_paths}:
        parser.error("Training and validation episode IDs overlap")
    if args.rendered_dir is None:
        train_data = ACTDataset(train_paths, args.chunk_size, augment=args.augment)
        val_data = ACTDataset(val_paths, args.chunk_size, train_data.norm_stats)
    batches_per_epoch = (len(train_data) + args.batch_size - 1) // args.batch_size
    if args.epochs is not None:
        args.steps = args.epochs * batches_per_epoch
    loader_options = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    if args.workers:
        loader_options.update(multiprocessing_context="spawn", prefetch_factor=1)

    # Fixed, evenly spaced samples keep evaluation comparable and inexpensive.
    eval_loaders = []
    import copy
    train_eval_data = copy.copy(train_data)
    train_eval_data.augment = False
    for data in (train_eval_data, val_data):
        indices = np.linspace(0, len(data) - 1, min(args.eval_samples, len(data)), dtype=int).tolist()
        eval_loaders.append(DataLoader(torch.utils.data.Subset(data, indices),
                                      batch_size=args.batch_size, **loader_options))
    policy = ACTPolicy(chunk_size=args.chunk_size).to(device)
    if initial_path is not None:
        expected = dict(chunk_size=args.chunk_size, hidden_dim=256, latent_dim=32)
        if initial["model_config"] != expected or initial["config"]["task"] != args.task:
            parser.error("Initialization checkpoint has a different model configuration or task")
        for key, value in train_data.norm_stats.items():
            if not np.allclose(value, initial["norm_stats"][key], rtol=1e-6, atol=1e-8):
                parser.error("Initialization checkpoint normalization differs from this training data")
        policy.load_state_dict(initial["model"])
        print(f"Loaded checkpoint step {initial['step']}", flush=True)
    if args.lr is None:
        args.lr = initial["config"]["lr"] if args.resume_checkpoint else 1e-4
    if args.augment is None:
        args.augment = initial["config"].get("augment", False) if args.resume_checkpoint else False
    train_data.augment = args.augment
    if args.freeze_batchnorm is None:
        args.freeze_batchnorm = initial["config"].get("freeze_batchnorm", False) if args.resume_checkpoint else False
    policy.configure_batchnorm(args.freeze_batchnorm)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    start_step, samples_seen = 0, 0
    if args.resume_checkpoint:
        optimizer.load_state_dict(initial["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if "sampler_seed" in initial:
            if initial["config"]["batch_size"] != args.batch_size:
                parser.error("Cursor resume requires the original batch size")
            if initial["train_episodes"] != [str(p.resolve()) for p in train_paths]:
                parser.error("Cursor resume requires identical ordered training episodes")
            if args.rendered_dir and initial["config"].get("rendered_manifest_sha256") != train_data.data_signature:
                parser.error("Cursor resume requires the identical rendered buffer manifest")
            args.seed = initial["sampler_seed"]
            start_step, samples_seen = initial["step"], initial["samples_seen"]
            torch.set_rng_state(initial["torch_rng_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(initial["cuda_rng_states"])
            print(f"Resuming at step {start_step}, sample cursor {samples_seen}", flush=True)
        else:
            print("Legacy checkpoint: restored optimizer, starting a NEW full data pass; "
                  "previous sample order was not saved.", flush=True)
    if start_step >= args.steps:
        parser.error("Requested run is already complete at this checkpoint")
    del initial
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    if args.rendered_dir:
        config["rendered_manifest_sha256"] = train_data.data_signature
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    abc_commit = subprocess.check_output(
        ["git", "-C", str(root / "abc"), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"Device: {device} | Task: {args.task}", flush=True)
    print(f"Train: {len(train_paths)} episodes, {len(train_data)} timesteps | "
          f"Validation: {len(val_paths)} episodes, {len(val_data)} timesteps", flush=True)
    print(f"Checkpoints: {args.output}", flush=True)
    print(f"Evaluation uses up to {args.eval_samples} fixed samples per split, z=0.", flush=True)
    print(f"Run: {args.steps} batches | {batches_per_epoch} batches per full pass", flush=True)
    best_val = float("inf")
    started = time.monotonic()
    def save_checkpoint(step, is_best):
        checkpoint = dict(
            model=policy.state_dict(), optimizer=optimizer.state_dict(), step=step,
            config=config, model_config=dict(chunk_size=args.chunk_size, hidden_dim=256, latent_dim=32),
            norm_stats={k: v.tolist() for k, v in train_data.norm_stats.items()},
            camera_keys=train_data.camera_keys, data_fps=30,
            camera_height=train_data.episodes[0][1].get("image_height", 224),
            camera_width=train_data.episodes[0][1].get("image_width", 224),
            camera_backend=train_data.episodes[0][1].get("render_backend", "mujoco"),
            train_episodes=[str(p.resolve()) for p in train_paths],
            val_episodes=[str(p.resolve()) for p in val_paths],
            abc_commit=abc_commit, best_val_l1=best_val,
            samples_seen=samples_seen, epochs=samples_seen / len(train_data),
            sampler_seed=args.seed, torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        )
        temporary = args.output / "checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "last.pt")
        if is_best:
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best.pt")

    if initial_path is not None:
        best_val = evaluate(policy, eval_loaders[1], device)
        print(f"Initial val z=0 L1 {best_val:.4f}", flush=True)
        save_checkpoint(start_step, True)
        with (args.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=start_step, val_l1=best_val,
                                   samples_seen=samples_seen, baseline=True,
                                   epochs=samples_seen / len(train_data),
                                   elapsed_seconds=time.monotonic() - started)) + "\n")
    iterator = iter(())
    policy.train()
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            # Fully exhausting a bounded loader shuts its decoder processes down.
            sampler = CursorBatchSampler(len(train_data), args.batch_size, args.seed,
                                         samples_seen, args.worker_recycle_every)
            loader = DataLoader(train_data, batch_sampler=sampler, **loader_options)
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        predictions, mu, logvar = policy(**batch)
        loss, l1, kl = compute_loss(predictions, batch["actions"], batch["is_pad"],
                                    mu, logvar, kl_weight=args.kl_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        samples_seen += len(batch["state"])
        if step == 1 or step % 10 == 0:
            print(f"Step {step:5d} | Loss {loss.item():.4f} | L1 {l1.item():.4f} | "
                  f"KL {kl.item():.4f} | Passes {samples_seen / len(train_data):.4f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            train_l1, val_l1 = [evaluate(policy, item, device) for item in eval_loaders]
            print(f"Eval {step}: train z=0 L1 {train_l1:.4f} | val z=0 L1 {val_l1:.4f}", flush=True)
            record = dict(step=step, train_l1=train_l1, val_l1=val_l1,
                          loss=loss.item(), l1=l1.item(), kl=kl.item(),
                          samples_seen=samples_seen, epochs=samples_seen / len(train_data),
                          elapsed_seconds=time.monotonic() - started)
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            improved = val_l1 < best_val
            best_val = min(best_val, val_l1)
            save_checkpoint(step, improved)
    print(f"Finished: {samples_seen} samples, {samples_seen / len(train_data):.6f} full passes", flush=True)


if __name__ == "__main__":
    main()
