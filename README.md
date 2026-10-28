# lp-capsule

Lightning Pose (LP 2.4.3) predict and train on Code Ocean GPUs, one config per
run, plus a launcher that fans predict batches out over many sessions. It
serves a fast label → train → predict → relabel loop with
lp-relabel (local, not yet published).

- **Worker** (`code/`): the App Panel has one field, `config`. `task: predict`
  runs a model on frame windows (or whole videos) of attached session assets;
  `task: train` trains or fine-tunes on an attached LP project.
- **Launcher** (`scripts/launch.py`, run on your Mac): resolves assets by name or
  id, starts parallel worker runs with those assets attached (nothing goes into
  the capsule's `datasets.json`), caps active runs, downloads predictions, and
  captures trained models as data assets.

Video timing stays in the existing libraries. Clips come from
`aind_dynamic_foraging_behavior_video_analysis.video_clips.cut_clip`, which
seeks by the MP4's own sample timestamps and gives frame-exact clips with even
timestamps. Notched whole videos are restamped by
`aind_video_utils.restamp_uniform`. Neither leaves the run: temporary files
live in `/results/.work-*` and are deleted. **Outputs are keyed by source frame
index and the capsule takes no times.** Behavior time is joined afterwards from
`video_timing_qc.correct_video_timing`'s `harp_time`.

## Configs

See `code/run_config.py` and `examples/`. Parsing is strict: unknown keys fail.

```yaml
task: predict
model: ephys_BottomCamera_Pylon2Test2   # mount name under /data
reader: dali                            # or opencv (about 0.5 px different; keep one fixed)
precision: fp16                         # or fp32
batch_size: 0                           # 0 = the model config's
videos:
  - asset: behavior_782394_2025-04-23_10-51-14
    camera: BottomCamera                # or path: <relative to the asset>
    windows: [[120000, 120500]]         # [start, end) source frames; omit = whole video
```

```yaml
task: train
project: pylon2test2_r01                # CollectedData.csv, labeled-data/, videos/
base_config: ephys_BottomCamera_Pylon2Test2   # a model asset, or <asset>/<path>.yaml
init_checkpoint: ephys_BottomCamera_Pylon2Test2   # optional: fine-tune
overrides: {training: {max_epochs: 100, min_epochs: 100}}
predict_videos: false
```

The `config` field takes YAML text, or a path to a YAML file (absolute, or
relative to `/data` or `code/`). The worker checks that every asset the config
names is mounted, so runs started by hand in the UI fail early and clearly.

## Outputs (`/results`)

- `run_config.yaml`: the config as run, with defaults filled in.
- `run.json`: package and ffmpeg versions, model files and checkpoint SHA-256,
  keypoint names and where they came from, reader, precision, per-window
  timings. `status` is written last: `ok`, or `failed` with the error. The
  launcher trusts `ok` only together with CO's end status and exit code.
- predict: `predictions/<asset>_<camera>.csv`, in LP's CSV format (scorer /
  bodyparts / coords header). Its index is the **source frame**, and it holds
  only the window rows.
- train: `model/`, a bundle predict reads as is (`config.yaml` with
  `keypoint_names` written in, the best checkpoint under `tb_logs/`, and LP's
  labeled-frame predictions), plus `pixel_error_summary.csv`. Pixel error is
  per set and keypoint, in pixels of the original labeled frames.

## Model bundles

Any asset layout works. The worker needs one LP config: a YAML with `model:` and
`data:`. Identical copies are fine. Different configs, or several different
checkpoints with no single `*-best.ckpt`, are an error. Keypoint names come from
the config's `data.keypoint_names`, or else from the `bodyparts` row of a
`CollectedData*.csv` in the asset (our LP 1.6 models leave the names null).

Models are loaded as in
[lightning-pose-inference-capsule](https://github.com/AllenNeuralDynamics/lightning-pose-inference-capsule):
the module is built from the checkpoint's hyperparameters, with no labeled data and
no backbone download. LP 1.6 `upsampling_layers.*` keys are renamed to
`head.upsampling_layers.*`, and the weights are loaded strictly. Only heatmap
models are supported.

Fine-tuning goes through LP's `model.checkpoint`, which LP loads with
`strict=False`, so a mismatched checkpoint would be dropped without any error.
The worker therefore first loads the init checkpoint strictly in its own
architecture. It checks that the architecture and keypoints match the new
config, and only then hands LP a re-saved copy.

## Launcher

```bash
uv run scripts/launch.py predict examples/predict_batch.yaml --out runs/r01 --dry-run
uv run scripts/launch.py predict examples/predict_batch.yaml --out runs/r01 --max-concurrent 10
uv run scripts/launch.py train examples/train.yaml --name pylon2test2_ft_r01
```

- The token comes from `API_SECRET`, or else from the Keychain item `codeocean-api`.
- Batch configs are predict configs whose assets can be names or ids.
  `--sessions-per-run N` groups sessions into runs.
- `<out>/manifest.json` is rewritten every pass. Launching again into the same
  `--out` skips finished runs, follows running ones and, with `--retry-failed`,
  relaunches failed ones.
- `--capture` also stores each predict run as an external asset under
  `s3://aind-scratch-data/matt.becker/lp/predictions/`. `train` always captures,
  to `.../lp/models/<name>/`, so the new model can be named in predict configs.
  The launcher refuses a `--name` that is already taken.

## Setting up on Code Ocean

1. Create the capsule from this repo with **Clone from Git**, not Copy.
2. Environment: the base image is `cuda12-base-uv:main-5692e50`. If no GPU machines
   are offered, select it in the Environment UI: CO tracks the base image by the
   `# hash:` line, not by `FROM`. Check that the Dockerfile still says
   `COPY git-ask-pass /`. `postInstall` fails the build on a broken import or an
   ffmpeg older than 5.1.
3. Pick a GPU machine, and set `CAPSULE_ID` in `scripts/launch.py` (done: CO capsule 9589758).
4. Register each model bundle as an external data asset named after its folder. Done so far:
   `ephys_BottomCamera_Pylon2Test2` (`547f5535-97b6-49b1-a26e-cc3d051ae881`), from
   `s3://aind-scratch-data/matt.becker/lightning_pose_test/ephys_BottomCamera_Pylon2Test2/`.

## Differences from aind-capsule-template

This repo was generated from
[aind-capsule-template](https://github.com/AllenNeuralDynamics/aind-capsule-template) and
keeps its layout, `.gitignore`, `code/run` and top-level `run()`. It differs where the
template's choices don't work for this capsule:

- **Base image** `cuda12-base-uv` (GPU, CUDA 12.8), not `mambaforge3`. Lightning Pose
  needs a GPU. `.codeocean/environment.json` turns on `post_install`, and
  `environment/postInstall` installs everything; the Dockerfile header is the template's.
- **Installs** are Lightning Pose's own (`apt install ffmpeg`, `pip install lightning-pose`),
  with LP pinned to 2.4.3: LP 2.4.2 gives centre-of-frame keypoints with kornia >= 0.9.
- **`code/run`** calls the image's venv Python (`/opt/venv`), not the `python` first on `PATH`.
- **Additions:** `.codeocean/app-panel.json` (one `config` field), `scripts/launch.py`,
  `tests/`, `examples/`, and `pyproject.toml` / `uv.lock` for the local dev environment.

## Development

```bash
uv sync && uv run pytest     # no GPU, torch or LP needed: a stand-in model
uv run ruff check . && uv run ruff format --check .
```

The tests cut real clips with ffmpeg and check that prediction rows land on
their source frames. They also cover every known bundle layout (Pylon2Test1/2,
ephys_bottom-camera, Anna's), config parsing, building the training config, and
the launcher's planning and resume logic. Anything that needs LP (model loading,
DALI, training) is checked on CO: the milestones in the plan.

## Not done yet

- Milestones 1–2 on CO: the environment build, then predictions checked against
  the scicomp full-session Pylon2Test2 CSVs. That check covers the shift from
  LP 1.6 vs 2.4, `cut_clip` re-encoding (CRF 18) and the reader, plus frame
  alignment on a session with dropped frames.
- DALI on very short windows (fewer frames than a batch) is untested.
- lp-relabel: write batch configs, and read sparse predictions indexed by
  source frame.
- Training's loss-curve plot (the TensorBoard logs are in `model/tb_logs`).
