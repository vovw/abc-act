# ACT bottle-task training record

Recorded on 2026-09-21 from saved checkpoints, run configurations, source code,
the rendered-data manifest, and measured rollout results. This describes the
policy that produced the successful video, rather than the latest CLI defaults
or the state of a separate ongoing training run.

The verified policy is `checkpoints/bottles-working/policy.pt`, copied from
`checkpoints/bottles-live-recovery/eval-step-2000.pt`. Its SHA-256 is
`e9da76b22b66ac5af6de6f17ac1d63cfd6b03ded18323b16e103a2fdaa76f784`.
The [machine-readable record](act-training-record.json) preserves the actual
checkpoint configurations, optimizer settings, normalization arrays, lineage,
episode lists for the rendered buffer, metrics, and verification results.

## What lives in act.py?

[`src/abc_act/policies/act.py`](../src/abc_act/policies/act.py) contains our ACT
implementation: both dataset loaders, image/state encoder, latent encoder,
action decoder, loss, training loop, checkpoint save/resume, offline validation,
simulator adapter, temporal ensembling, browser viewer, and MP4 recording.

It is the main entrypoint, but it is **not a standalone copy of everything used**.

| File or directory | What it supplies |
| --- | --- |
| `src/abc_act/policies/act.py` | Core ACT model, training, evaluation, and inference |
| `abc/abc_minimal/episode_io.py`, `dataloader.py`, `preprocess.py` | Episode loading, video decoding, image transforms |
| `abc/abc_sim/` and its model assets | Simulator, task definition, cameras, robot, and physics |
| `pyproject.toml`, `uv.lock`, root `.venv` | Installed environment and dependency versions |
| `abc/cache/train_sim`, `abc/cache/val_sim` | Original demonstration data |
| `outputs/act-live-training-data/` | Rendered fine-tuning buffer and its exact sample manifest |
| `outputs/act-bn-ablation/original_stats.pt` | Prepared initialization checkpoint for the successful fine-tune |
| `checkpoints/bottles-working/policy.pt` | Learned weights, normalization, model dimensions, camera order, and training state |
| `outputs/act-policy-success/` | Actual ACT video, rollout JSON, and verification hashes |

Two preparation scripts ran outside `act.py`. Historical copies are now saved
in [act-training-support](act-training-support/README.md): the rendered-buffer
generator and the BatchNorm ablation/preparation script. These steps are not
automatically performed by invoking the trainer.

`outputs/` and `checkpoints/` are Git-ignored. Committing the code/docs alone
does not preserve the weights, data, or videos. Inference does not need the
training buffer or preparation scripts; it needs the code, environment, simulator
assets, and working checkpoint. Repeating the training needs the data and its
initialization checkpoints too.

## Architecture and inputs

| Parameter | Value used |
| --- | --- |
| Task | `sim_put_the_plastic_bottles_in_the_bin` |
| State / action dimension | 14 / 14; position commands in the dataset convention |
| Cameras, in order | `top`, `left`, `right` |
| Raw simulation images | RGB, 168 × 224 per camera |
| Model image input | Aspect-preserving resize and zero pad to 224 × 224 |
| Image normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`; helper adds `1e-6` to std |
| Backbone | Shared ImageNet-pretrained ResNet18; remove average pooling/classifier |
| Visual projection | 1 × 1 convolution, 512 → 256 channels |
| Image tokens | 3 × 7 × 7 = 147 tokens, each 256-dimensional |
| Visual position embeddings | Learned camera, row, and column embeddings |
| State token | Linear projection 14 → 256; 148 observation tokens including images |
| Hidden dimension | 256 |
| Action chunk size | 20; output shape `[batch, 20, 14]` |
| Latent dimension | 32 |
| CVAE encoder | 2 Transformer encoder layers; summary + state + 20 action tokens |
| Action decoder | 2 Transformer decoder layers; 20 learned action queries |
| Attention heads | 8 in both Transformers |
| Feedforward dimension | 1024 in both Transformers |
| Transformer dropout / activation | 0.1 / ReLU |
| Transformer normalization | Default post-norm, LayerNorm epsilon `1e-5` |
| Decoder memory | Projected latent token + state token + 147 image tokens |
| Output head | Linear projection 256 → 14 |
| Extra observation Transformer encoder | None in this implementation |

State and action statistics are computed per dimension over the original
training split using population standard deviation, floored at `1e-6`.
Validation and rendered-data fine-tuning reuse those training statistics:

```text
normalized_state  = (state - state_mean) / state_std
normalized_action = (action - action_mean) / action_std
commanded_action  = predicted_normalized_action * action_std + action_mean
```

Each sample at timestep `t` targets `actions[t:t+20]` within that episode.
Near the end of an episode, missing targets are zero-padded after normalization
and marked by `is_pad=True`. Padding is excluded from both the latent encoder's
attention and the reconstruction loss.

## Training objective and optimizer

```text
z = mu + exp(0.5 * logvar) * epsilon,  epsilon ~ N(0, I)

L1 = sum over valid action scalars of abs(prediction - target)
     / number of valid action scalars

KL = batch_mean(-0.5 * sum over 32 latent dimensions
                (1 + logvar - mu^2 - exp(logvar)))

loss = L1 + 10 * KL
```

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, one parameter group; no separate backbone learning rate |
| Betas | `(0.9, 0.999)` |
| Epsilon | `1e-8` |
| Weight decay | `0.01` |
| KL weight | `10.0` |
| Gradient clipping | Global norm 1.0 |
| Learning-rate scheduler / warmup | None |
| Mixed precision / gradient accumulation | None; float32, one optimizer update per batch |
| Training seed | 42 for Torch and NumPy |
| Device used | CUDA on RTX 4090 |
| CPU Torch threads | 4 |

Validation uses `model.eval()` and `z=0`, without passing demonstrated actions
to the policy. Offline L1 is measured in normalized action units. It is a
different measurement from simulator task success.

## Exact training lineage of the working policy

These are the stages inherited by the published weights. The selected step is
the checkpoint used by the next stage; it is not necessarily the end of that run.

| Stage | Selected checkpoint / step | Batch | LR | Image augmentation | BN frozen | Eval interval / samples per split |
| --- | --- | --- | --- | --- | --- | --- |
| Initial simulation training | `bottles-sim-10000/best.pt`, 9,000 | 8 | `1e-4` | No | No | 500 / 256 |
| Longer original-data training | `bottles-sim-full-pass/best.pt`, 260,000 | 8 | `3e-5` | No | No | 5,000 / 256 |
| Recovery of original-data training | `bottles-sim-recovery/eval-step-5000.pt`, 5,000 new steps | 8 | `1e-5` | No | No | 5,000 / 512 |
| Image-augmentation fine-tuning | `outputs/act-bn-ablation/augmented_stats.pt`, 5,000 | 8 | `1e-5` | Yes | No | 2,500 / 512 |
| BatchNorm preparation | `outputs/act-bn-ablation/original_stats.pt` | — | — | — | — | No training updates |
| Live-rendered fine-tuning | `bottles-live-recovery/eval-step-2000.pt`, 2,000 | 16 | `1e-5` | No | Yes | 1,000 / 512 |

Checkpoint paths in the table are under `checkpoints/` unless prefixed with
`outputs/`. All training stages use chunk size 20, KL weight 10, and seed 42.
The first two stages used four loader workers. The recovery, augmentation, and
rendered stages used two workers, recreated every 2,000 batches, with spawn
and prefetch factor 1. Worker recycling was added after a decoder worker
exhausted host memory.

Each `--init-checkpoint` stage starts a new optimizer. The original-data recovery
stage instead restored the optimizer from step 260,000 and changed its LR to
`1e-5`. That legacy checkpoint lacked a sample cursor, so the recovery stage
started a **new shuffled pass**, rather than continuing at a known unseen row.

The BatchNorm preparation kept the augmentation-step-5,000 learned weights,
but replaced 60 `running_mean`, `running_var`, and `num_batches_tracked` buffers
with the corresponding values from recovery step 5,000. It did not restore the
BN affine weights. The final fine-tune froze both the resulting BN running
statistics and affine parameters; convolution weights continued learning.
Its optimizer was newly initialized, so the optimizer stored in the prepared
checkpoint was not used.

The augmentation run eventually reached 10,000 steps, but those later weights
are not ancestors of this working policy. The separate
`bottles-sim-frozen-bn` experiment is also not in its lineage. A prepared
100-action-chunk candidate was not used.

## Original data and rendered fine-tuning data

The original split has **5,588 training episodes / 5,037,922 timesteps**, plus
**287 validation episodes / 266,541 timesteps**. One full training pass at
batch size 8 takes **629,741 batches**, including the final partial batch.
Overlapping action targets do not count as extra timestep coverage.

The working policy does not demonstrate completion of that full pass. Its
inherited original-data stages used 72,000 + 2,080,000 + 40,000 + 40,000 sample
presentations. These stages restart sampling, so their sum is not a count of
unique timesteps. Later progress in the separate original-data run is not
incorporated automatically into `bottles-working/policy.pt`.

| Rendered-buffer parameter | Value |
| --- | --- |
| Directory | `outputs/act-live-training-data` |
| Renderer | Classic MuJoCo |
| Data-generation RNG seed | 20260921 |
| Training selection | 64 evenly spaced training episodes, 128 evenly spaced timesteps each |
| Training observations | 8,192 |
| Validation selection | 16 separate validation episodes, 32 evenly spaced timesteps each |
| Validation observations | 512 |
| Reserved rollout scenes | Three diagnostic validation scenes excluded from rendered validation; already disjoint from training |
| Training perturbation probability | 0.5; actual perturbed observations: 4,064 / 8,192 |
| Robot joint noise | Gaussian standard deviation 0.01 radians |
| Gripper noise | Gaussian standard deviation 0.02 in policy units, clipped to `[0, 1]` |
| Validation perturbations | None |
| Scene setup per observation | Restore demonstration scene pose at `t`, zero velocity, optionally perturb robot pose, then render |
| Training targets | Original demonstrated future actions; targets are not recomputed after perturbation |
| Normalization | Reuse full original-training statistics from initialization checkpoint |
| Image augmentation during final fine-tune | Off |
| Buffer manifest SHA-256 | `6010c257d946d56c57aa75eaf8d44131b9764bb05f04906b5a93aed0e63d9f1b` |

The buffer is rendered around demonstrated states; it is not an on-policy
collection of ACT failures with newly labeled corrective actions.

The earlier augmentation stage used the upstream image transform: top-camera
rotation uniformly between -5 and +5 degrees, a random crop retaining 95% of
height and width followed by resizing, and per-camera brightness `[0.7, 1.3]`,
contrast `[0.6, 1.4]`, and saturation `[0.5, 1.5]` multipliers. Validation did
not use these augmentations.

## Final fine-tune and checkpoint selection

The historical fine-tune ran for 5,000 updates. The published checkpoint is
from update **2,000**, after **32,000 sample presentations / 3.90625 passes over
the 8,192-observation buffer**. These are buffer passes, not original-data passes.

| Fine-tune step | Train z=0 L1 | Validation z=0 L1 | Buffer passes |
| --- | --- | --- | --- |
| 0 | — | 0.150378 | 0 |
| 1,000 | 0.095186 | 0.121163 | 1.953125 |
| **2,000 — published policy** | **0.084308** | **0.123490** | **3.90625** |
| 5,000 | 0.064070 | 0.130148 | 9.765625 |

`best.pt` means lowest sampled offline validation L1, which occurred at step
1,000. Step 2,000 was retained after actual rollout successes. Those are two
different checkpoint-selection criteria. The model's KL term at step 2,000
was approximately `1.14e-7` on the logged training batch.

With the existing prepared checkpoint and buffer, this command repeats the
final stage through the selected 2,000-update point. Run from the repository
root; the output directory must be new:

```bash
source .venv/bin/activate
python -m abc_act.policies.act --device cuda \
  --task sim_put_the_plastic_bottles_in_the_bin \
  --rendered-dir outputs/act-live-training-data \
  --init-checkpoint outputs/act-bn-ablation/original_stats.pt \
  --steps 2000 --batch-size 16 --chunk-size 20 \
  --lr 1e-5 --kl-weight 10 --seed 42 \
  --no-augment --freeze-batchnorm \
  --workers 2 --worker-recycle-every 2000 \
  --eval-every 1000 --eval-samples 512 \
  --output checkpoints/bottles-live-reproduction
```

The endpoint is `checkpoints/bottles-live-reproduction/last.pt`; `best.pt` may
point to an earlier step. This recipe reproduces the configuration, not a
guarantee of identical weights or contact-sensitive rollout outcomes. For the
historical 5,000-update schedule, change `--steps` to 5000. The trainer only
automatically saves `last.pt` and `best.pt`; the named `eval-step-*.pt` files
used in experiments were separately retained snapshots.

The saved final-stage config contains default `train_real`/`val_real` paths
and `execute_actions=5`. Neither describes the successful experiment: with
`--rendered-dir`, the manifest selects the data; rollout flags select action
execution. No real-data training was used by this final stage.

## Time spent on the successful checkpoint's training stages

The inherited training stages took approximately **two hours** on the RTX
4090, including their periodic offline evaluations. This is a sum of the
stages leading to the selected weights, not the duration of all experiments
or the total elapsed project time.

| Stage through its selected checkpoint | Recorded elapsed time |
| --- | --- |
| Initial simulation training, step 9,000 | Approximately 5 minutes from config/checkpoint file timestamps; no elapsed field in that older metrics file |
| Longer original-data training, step 260,000 | 1 hour 40 minutes 8 seconds |
| Recovery, step 5,000 | 5 minutes 19 seconds |
| Augmentation, step 5,000 | 6 minutes 41 seconds |
| Final rendered fine-tune, step 2,000 | 1 minute 48 seconds |

The four stages with explicit elapsed metrics sum to 1 hour 53 minutes
57 seconds, plus the initial stage. Generating the rendered training and
validation buffer took another approximately 4 minutes 50 seconds. Downloads,
startup outside the training timers, BN preparation, rollout evaluation,
debugging, discarded experiments, and later unused training are additional.
The final 1-minute-48-second fine-tune depends on all the earlier trained weights;
it is not training from scratch.

## Successful inference settings and measured limits

| Setting | Published video |
| --- | --- |
| Policy mode | Evaluation, latent `z=0`; only current images/state as observations |
| Renderer | Classic MuJoCo, 168 × 224 camera frames |
| Control rate | 30 Hz |
| Physics | 17 substeps per action, timestep `1 / (30 * 17)` seconds |
| Action execution | Execute all 20 actions, then observe and predict again (~0.667 seconds per chunk) |
| Temporal ensembling | Off |
| Scene | Fresh randomized seed 1; four bottles; no saved demonstration scene |
| Episode limit | 2,100 actions |
| Success requirement | 90 consecutive successful actions while ACT continues controlling |
| Result | All four bottles in bin; 684 actions, 22.8-second video |

```bash
python -m abc_act.policies.act --sim --headless --device cuda \
  --checkpoint checkpoints/bottles-working/policy.pt \
  --camera-backend mujoco --seed 1 \
  --execute-actions 20 --sim-steps 2100 --success-hold-steps 90 \
  --video outputs/act-reproduction.mp4 \
  --rollout-json outputs/act-reproduction.json
```

For the interactive viewer, remove `--headless`, `--video`, and `--rollout-json`,
then open `http://127.0.0.1:8080` and press Play.

On the small fresh-seed test using 20-action execution, seeds 1 and 42 succeeded;
seed 0, with six bottles, failed. These three seeds were used during model
development, not a large independent benchmark. The evidence supports actual
task completion in tested scenes, not broad reliability. The combined recipe
worked; the independent causal contribution of every change has not been
isolated.

## Environment and evidence

Checkpoint metadata records upstream ABC commit
`6c467cebcecf16a4dce79e6fd87a7ca2281c3ef0`. The project pins PyTorch
`2.11.0+cu128` and torchvision `0.26.0+cu128`; the root environment uses Python
3.12. Consult `uv.lock` for the resolved dependencies. Source hashes at the
time of documentation are in `act-training-record.json`; they describe the
current implementation, not a historical source snapshot of every run.

Local evidence: each run's `config.json` and `metrics.jsonl`, the checkpoints
listed above, `outputs/act-bn-ablation/results.json`, the rendered manifest,
and `outputs/act-policy-success/{rollout,verification}.json`. The successful
video is `outputs/act-policy-success/act-success.mp4`.
