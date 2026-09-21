# Historical preparation scripts

These copies preserve the two preparation steps performed outside `act.py`.
They were archived while documenting the successful policy; copying them did
not regenerate data, change checkpoints, or start training.

- `generate_rendered_data.py` is an exact copy of
  `outputs/act-live-training-data/generate.py`. It renders the 8,192 training
  and 512 validation observations. It has the original absolute project path,
  requires the original simulator dataset/assets, reads reserved episode IDs
  from `outputs/act_ground_truth_replay.json`, and requires its output directory
  not to exist. `reserved_rollout_scenes.json` preserves that small input report.
- `bn_ablation_original.py` is an exact copy of `/tmp/act_bn_ablation.py`. It
  compares augmentation-run BatchNorm statistics with pre-augmentation
  statistics and writes the prepared initialization checkpoint.

These are historical scripts, not portable command-line tools. In particular,
**do not rerun the BN script unchanged to reconstruct the working policy**:
it read `bottles-sim-augmentation/last.pt` when that file was at step 5,000.
The run subsequently advanced to step 10,000. The original step-5,000 source
was preserved at `outputs/act-bn-ablation/augmented_stats.pt`. For a new
reconstruction, use that fixed source, verify `checkpoint['step'] == 5000`,
and write to a fresh output directory. The pre-augmentation statistics source
remains `checkpoints/bottles-sim-recovery/eval-step-5000.pt`.

The BN preparation replaces only model keys ending in `running_mean`,
`running_var`, or `num_batches_tracked`. The resulting checkpoint is for
`--init-checkpoint`, which starts a fresh optimizer; its saved optimizer was
not updated to reflect the buffer replacement.

See [the training record](../act-training.md) for hyperparameters, lineage,
the existing prepared checkpoint, and commands to repeat the final fine-tune.
