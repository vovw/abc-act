# abc-act

Imitation learning experiments using ABC data and environments, with ACT as
the first policy. The single-file ACT implementation includes episode loading,
normalization, training, validation, and checkpoints.

## Structure

```text
abc/                         # pinned upstream Git submodule
src/abc_act/
  data/                      # episode loading, normalization, action chunks
  policies/                  # ACT first; add other policies here later
  training/                  # optimization, validation, checkpoints
  evaluation/                # ABC environment adapters and rollout metrics
data/                        # local datasets (ignored by Git)
outputs/                     # experiment logs and results (ignored by Git)
checkpoints/                 # model weights (ignored by Git)
```

Keep experiment code in `src/abc_act/`. Reuse ABC's episode readers and
environment APIs through small adapters. The upstream repository ships
ABC-DiT and ABC-VLA; its training entrypoint does not train ACT.

The current learning implementation lives entirely in `policies/act.py`.
The other package directories are placeholders for later organization.

See the [ACT training record](docs/act-training.md) for the successful bottle
policy's exact hyperparameters, checkpoint lineage, data preparation, and
training/inference commands. The core implementation is in `act.py`; the
record also identifies the upstream dependencies and preparation scripts
used outside that file.

## Setup

For a fresh clone, use `git clone --recurse-submodules <this-repository-url>`.
For an existing clone:

```bash
git submodule update --init --recursive
uv sync
uv run abc-act
```

The root `.venv` installs this project and the local `abc-minimal` package
in editable mode, plus PyTorch, torchvision, and ABC's dependencies. Python
source edits take effect without reinstalling. The Linux GPU wheels match
ABC's CUDA 12.8 PyTorch pin.

Activate the root environment once in each terminal (even if you previously
activated `abc/.venv`):

```bash
source /home/sra/ksagar/abc-act/.venv/bin/activate
python -m abc_act.policies.act
```

After activation, that module command works from any directory. For an
editor's Python interpreter, select the root `.venv/bin/python`. Without
activation, use `uv run python -m abc_act.policies.act` from the project
root or any directory under `src/abc_act`.

These commands execute your current `act.py`; unfinished code can still
raise errors. Relative data paths resolve against the terminal's working
directory. Use an absolute dataset path or resolve a project-relative path
against `Path(__file__).resolve().parents[3]` inside `policies/act.py`.

To download the preview data using the same environment:

```bash
cd abc
../.venv/bin/python prepare.py
```

The `abc/` submodule has its own uv project; running `uv run` there selects
its separate environment. Use the activated root Python for a consistent
environment. See [ABC's setup](abc/README.md) for upstream tools.

## Train one task

With the root environment activated, run from any directory:

```bash
python -m abc_act.policies.act --device cuda --steps 1000
```

This selects only `put_the_plastic_bottles_in_the_bin` episodes from the
preview's `train_real` and `val_real` pools. It shuffles all training timesteps
and cycles through the loader as needed, keeping action chunks within each
episode. The separate throwing task is excluded. Normalization statistics
are computed across training episodes and reused for validation.

The default run uses batch size 8, action chunk size 20, and KL weight 10.
Every 100 steps it evaluates up to 128 fixed, evenly spaced timesteps from
each split with `z=0` (no target actions passed to the model). These sampled
normalized L1 metrics measure offline action error, not rollout success.
Use `--eval-samples` to increase coverage. `--steps`, `--batch-size`,
`--workers`, `--lr`, `--kl-weight`, `--task`, `--train-dir`, and `--val-dir`
are configurable; run with `--help` for details.

Each run creates a new timestamped directory under `checkpoints/`, or a new
directory specified with `--output`. It saves `config.json`, `metrics.jsonl`,
`last.pt`, and `best.pt` (lowest sampled validation L1). Checkpoints include
model and optimizer weights, model dimensions, normalization, camera order,
episode paths, seed/config, step, and the upstream ABC commit. Use
`--resume-checkpoint` to restore the optimizer and saved sample cursor into a
new output directory. `--steps`/`--epochs` set the total target for that cursor.
Older checkpoints lack a sample cursor: they restore weights and optimizer but
start a new shuffled pass, which is explicitly reported in the log.

For more episodes, use ABC's full task download from its directory with the
root environment's Python (the upstream README estimates about 35 GB):

```bash
cd abc
../.venv/bin/python prepare.py --full
```

Dataset download is separate from training. The trainer automatically picks
up additional episodes whose metadata matches the selected task.

## Train on bottle-placement simulation data

Download the task-specific release (about 14.1 GB; 5,588 training episodes and
287 validation episodes):

```bash
cd abc
../.venv/bin/python prepare.py --sim-data sim_put_the_plastic_bottles_in_the_bin
cd ..
python -m abc_act.policies.act --device cuda \
  --task sim_put_the_plastic_bottles_in_the_bin \
  --train-dir abc/cache/train_sim --val-dir abc/cache/val_sim \
  --steps 10000 --eval-every 500 --eval-samples 256 --workers 4
```

The loader pads rectangular camera frames to 224×224 without stretching them,
then applies ImageNet normalization. Simulation checkpoints record source
camera dimensions and renderer so the viewer can reproduce their inputs.

Use `--epochs 1` to visit every training timestep once, including the last
partial batch; this overrides `--steps`. The bottle simulation split has
5,037,922 timesteps, requiring 629,741 batches at batch size 8. Each sample
contains current images/state plus up to 20 future actions; overlapping action
chunks do not count as separate passes through the dataset.

`--init-checkpoint checkpoints/bottles-sim-10000/best.pt` initializes learned
weights for a new run, checks model/task/normalization compatibility, and
creates a new optimizer and sampling order. It is not an exact mid-epoch
resume. Logs and checkpoints record samples seen and passes in the new run.

Video decoder workers are recreated every `--worker-recycle-every` batches
(default 1000), limiting the memory growth that interrupted the earlier run.
Recycling does not repeat or skip samples. The starting checkpoint is evaluated
and saved as the initial best, so further training cannot silently replace it
with a worse checkpoint by the selected offline metric.

`--augment` enables ABC's image augmentation for training only: top-camera
rotation/cropping and camera brightness, contrast, and saturation changes.
Validation and the fixed training evaluation samples remain unaugmented. This
is being tested against the measured sensitivity to live camera appearance;
it is not yet a demonstrated task-success improvement.
`--freeze-batchnorm` fixes backbone normalization statistics and affine
parameters while allowing convolution weights to learn. This avoids shifting
the normalization as augmented image colors vary. Resume inherits both flags;
`--no-augment` and `--no-freeze-batchnorm` explicitly disable them.

## Watch ACT in the simulator

The locally verified checkpoint is `checkpoints/bottles-working/policy.pt`.
It completed a fresh four-bottle scene (seed 1), then maintained success for
90 more actions. The actual ACT inference video is
`outputs/act-policy-success/act-success.mp4` (22.8 seconds), with metrics and
checkpoint/video hashes beside it. Using 20-action chunks, fresh seeds 1 and 42
succeeded and seed 0 failed; this small test does not establish broad reliability.

From the project root, with the root environment activated:

```bash
python -m abc_act.policies.act --sim --device cuda \
  --checkpoint checkpoints/bottles-working/policy.pt \
  --seed 1 --execute-actions 20 --success-hold-steps 90
```

Open http://127.0.0.1:8080 and enable **Play**. The browser shows the 3D robot,
policy camera feeds, reward, and success status. Use **Reset with next seed**
for a new scene and **Playback speed** to adjust viewing speed. Ctrl+C in the
launching terminal stops the server. Choose another checkpoint path as needed;
use an absolute path when running outside the project root.

This adapter currently supports the put-bottles task. It restores checkpoint
normalization, predicts with `z=0`, denormalizes the 14 commanded positions,
and executes the full predicted chunk before observing again (20 actions for
the working checkpoint). `--execute-actions` changes
that interval; `--sim-steps` changes the default 2100-action episode limit.
Physics uses 17 substeps per action at the checkpoint's 30 Hz data rate.
The viewer uses the camera dimensions saved in the checkpoint. For the bottle
simulation release it defaults to classic MuJoCo: a saved-scene pixel comparison
matched its recorded images much more closely than the installed MJWarp
renderer, despite the dataset metadata naming MJWarp. `--camera-backend` can
override this choice. CUDA inference requires GPU access.

To evaluate one seed without opening a viewer:

```bash
python -m abc_act.policies.act --sim --device cuda --headless \
  --checkpoint checkpoints/bottles-working/policy.pt --seed 42
```

Rollout reward and success are printed in the terminal. Real-camera checkpoints
face an additional visual/environment transfer gap. Rollout task success is
distinct from offline validation L1 for either training domain.

Record an actual policy rollout and machine-readable results:

```bash
python -m abc_act.policies.act --sim --headless --device cuda \
  --checkpoint checkpoints/bottles-sim-recovery/best.pt \
  --execute-actions 20 --seed 42 \
  --video outputs/act-rollout.mp4 --rollout-json outputs/act-rollout.json
```

`--sim-episode /absolute/path/to/episode` uses that episode's scene and starting
state for a repeatable test or interactive viewer. All subsequent actions come from ACT;
recorded demonstration actions are not executed. `--temporal-ensemble 0.01`
instead replans every action and averages overlapping chunk predictions. Treat
the action-execution settings as experiments and compare measured task success.
The JSON records the checkpoint hash, seed, renderer, action settings, reward,
and success; an MP4 alone is not evidence of task completion.
`--success-hold-steps 90` continues policy control until the task evaluator
reports success for 90 consecutive actions (three seconds at 30 Hz).

The recovery experiment uses `--rendered-dir outputs/act-live-training-data`
with `--init-checkpoint` to retain the pretrained normalization. It contains
8192 live-rendered frames from 64 training episodes, with small joint
perturbations in about half, plus 512 unperturbed validation frames from 16
separate episodes. The three saved rollout-test scenes are excluded from both
buffer splits. Buffer passes are distinct from passes over the full original
5,037,922-timestep dataset; that longer run remains separate.

## Submodule workflow

The parent repository records an exact ABC commit. After checking out a
different upstream version inside `abc/`, run `git add abc` and commit the
updated pointer in this repository. To restore the recorded version, run
`git submodule update --init --recursive`.
