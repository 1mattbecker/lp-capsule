# exp1 results: bottom-camera LP training (2026-10-29 to 2026-10-30)

Eight models trained on combinations of Pylon2Test2 (ephys), round-1 (ephys) and
FIP labels, scored on four held-out test sets. ID = frames from mice seen in
training; OOD = two held-out mice per domain. Metric: median pixel error at the
last checkpoint (original frame pixels). Raw tables: `runs/exp1/eval/compare/`
(local, from `compare.py`).

## Answers

| Question | Answer |
|---|---|
| Does adding FIP hurt ephys? | **No.** Ephys error 3.53 to 3.74 (ID), 4.11 to 4.39 (OOD), within seed-to-seed noise. |
| Does adding ephys hurt FIP? | **No, it helps.** FIP OOD 4.05 to 3.22; the FIP-only model fails on paws. |
| Do models generalize across ephys vs FIP sessions without labels from both? | **No.** Ephys-only on FIP frames: nose 6-7 px vs 2-3, tongue 10 px vs 4-6. FIP-only on ephys frames: about 7 px vs about 4. Training on both covers both at no cost. |
| ID vs OOD (unseen mice) | OOD about 0.5-1 px worse for every model; no training choice closes the gap. |
| From scratch vs fine-tune | **Fine-tuning Pylon2Test2 on all data is best or tied** (FIP 2.31 / 2.50 px), 3x cheaper (1.4 h vs 4.1 h on Flex). |
| Best vs last checkpoint | Within about 0.5 px either way; the last epoch is fine. |
| Temporal context model | **Small gain:** about 0.35 px ephys, 0.2 px FIP OOD (beyond seed noise), at 2.7x training time on a 22 GB machine. |
| Semi-supervised losses | **Unanswered.** The run never learned (flat supervised loss, 50-130 px even on training frames); the fault is in its unlabeled-video input, not the losses. |

## Median error by model

| Model | Ephys ID | Ephys OOD | FIP ID | FIP OOD |
|---|---|---|---|---|
| Pylon2Test2 (old) | 4.54 | 4.82 | 2.70* | 2.32* |
| r01b (old fine-tune) | 4.35 | 4.48 | 3.31* | 2.58* |
| ephys only | 3.53 | 4.11 | 4.51 | 4.94 |
| FIP only | 6.86 | 7.01 | 2.48 | 4.05 |
| all data | 3.74 | 4.39 | 2.48 | 3.22 |
| all data, second seed | 3.71 | 4.32 | 2.62 | 3.27 |
| all data + context | 3.38 | 4.00 | 2.44 | 3.02 |
| all data + semi-supervised | 87.5 (failed) | 75.8 | 112.8 | 130.3 |
| fine-tune Pylon2Test2, ephys | 3.62 | 3.81 | 3.91 | 3.41 |
| **fine-tune Pylon2Test2, all data** | 3.53 | 4.24 | **2.31** | **2.50** |

\* Inflated: many FIP labels kept Pylon2Test2's prefilled predictions unchanged.
Treat differences under about 0.3 px as noise (two seeds differ by up to 0.14 px).

## Per keypoint, FIP OOD (unseen mice; no paws labeled)

| Model | nose | jaw | tongue L | tongue C | tongue R | spout L | spout R |
|---|---|---|---|---|---|---|---|
| ephys only | 6.7 | 3.7 | 5.1 | 10.5 | 6.2 | 2.1 | 5.3 |
| FIP only | 3.2 | 4.6 | 5.0 | 7.6 | 11.6 | 2.3 | 3.0 |
| all data | 3.0 | 3.5 | 4.8 | 8.2 | 6.9 | 2.3 | 2.6 |
| all + context | 3.1 | 2.9 | 4.6 | 6.2 | 8.1 | 2.3 | 2.7 |
| fine-tune, all | 2.3 | 2.4 | 3.9 | 5.8 | 8.9 | 1.5 | 2.0 |

Nose, jaw and spouts are 2-3.5 px; the tongue (6-9 px on unseen FIP mice) is
the weakest keypoint for every model. Paws have 0-13 labeled test frames per
set, too few to judge.

## Caveats

- Rigs: the FIP OOD mice ran on rigs that are also in training, so an unseen rig was never tested.
- Ephys OOD is 2 mice (32 frames).
- Test frames were chosen as hard (low model confidence), not at random.

## Next, toward full LP (unsupervised losses + context, each proven worth it)

1. Fix the unlabeled-video input and rerun the semi-supervised model.
2. Fine-tune with context and unsupervised losses from Pylon2Test2, against the plain fine-tune.
3. A fairer test set: random frames, labeled without prefills, more unseen mice, at least one held-out rig.
4. A tongue labeling round.

## Runs

Configs: `train.yaml` (6 runs, Flex T4), `train_video.yaml` (context and
semi-supervised, g6.2xlarge L4: both exceed 15 GiB), `eval.yaml` (18 runs).
Data: `build_data.py` (splits, FIP clips), `build_p2t2_clean.py` (Pylon2Test2
clips re-encoded and neighbour frames written with aind-video-utils).
