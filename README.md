# gray

Egocentric image and text in, body motion out, with a comparator that notices when the world
differs from what was predicted.

**Taking over the project:** start with [`HANDOFF.md`](HANDOFF.md).
**Overview with pictures:** open [`docs/index.html`](docs/index.html) in a browser after cloning (the Status tab first).
**Results, short:** [`STATUS.md`](STATUS.md). **What is open:** [`BACKLOG.md`](BACKLOG.md).
**Every experiment, in order:** [`DESIGN.md`](DESIGN.md).

This file only says which script does what. Computation runs on Viper: Python
`/u/enazaren/envs/torch/bin/python`, outputs under `/ptmp/enazaren/gray`. Trainings are submitted
with `jobs/submit.sh`, one-off scripts with `jobs/once.sbatch` (see `HANDOFF.md`, section 4, and
`jobs/clusters/README.md`); the older per-script job files in `jobs/` name their log and output at the top.

## The chain on moments

The head-path model predicts where the head goes from text and what the camera saw; the body
model generates the body along it. A sample ("moment") starts where a narration sentence starts.

| Step | Code | Job | Output in `/ptmp/enazaren/gray` |
|---|---|---|---|
| Pinhole image features, one vector per frame | `gray/pinhole.py` (lens: `gray/rectify.py`) | `pinhole.sbatch` | `pinhole/<id>/dino.npy` |
| Image grid 7 x 7 for frames that can be "now" | `gray/patches.py` | `patches.sbatch` | `patches/<id>/` |
| Room maps from the annotated walls and furniture | `gray/layout.py` | `layout.sbatch` | `layout/<id>.npz` |
| Free and blocked space per location, without the furniture boxes | `gray/ground.py` | `ground.sbatch` | `ground/<location>.npz` |
| Head path of each recording as one track | `gray/tracks.py` | `moments.sbatch` | `tracks/<id>.npz` |
| List of moments, navigation subset, data loader | `gray/moments.py` | `moments.sbatch` | `moments/moments.npz` |
| Model and training | `gray/head.py` | `head.sbatch` | `runs/<name>/` |
| Scoring against maps and text | `gray/score.py` | `head.sbatch`, `score.sbatch` | `runs/<name>/score.json` |
| The comparator: the image feature along the path, with its uncertainty | `gray/compare.py` | `compare.sbatch` | `runs/<name>/` |
| Body motion of each recording as one track (encoded again from the motion capture) | `gray/body.py` | `body.sbatch` | `body/<id>.npz` |
| The body model: the head path of the next 5 s -> the body along it | `gray/motion.py` | `motion.sbatch` | `runs/<name>/` |
| Checks of the body model: sampling settings, the chain with the head-path model, rolling forward | `gray/chain.py` | `chain.sbatch` | `runs/<name>/chain.json` |
| Prompt the chain at a spot in a recorded home (worker for the Loc_08 viewer; `--demo` without it) | `gray/scene_worker.py` | `scene.sbatch` | `runs/scene-demo/` |
| Workstation: serve the Loc_08 viewer and forward prompts to the worker | `gray/scene_server.py` | | |
| Head paths to judge by eye: per held-out sample the view, colour-coded plans, other prompts, other noise (cluster or workstation) | `gray/gallery.py` | `gallery.sbatch` | `runs/<name>/gallery/` |
| HumanML3D in the form of our samples: body latents, root path, and a head path made from the head joint; counts of candidate subsets | `gray/hml.py` | | `hml3d/hml.npz` |
| Workstation: videos of generated motion on the room map | `gray/animate.py` | | |
| Workstation: a video with the views the models were given, the body as a mesh in the home's scene, and the planned camera's view (Blender) | `gray/film.py`, `gray/film_blender.py`, `gray/seen.py` | | |
| For every frame a later frame from nearly the same pose (targets for the comparator) | `gray/revisits.py` | `revisits.sbatch` | `revisits/<id>.npy` |
| Does the comparator notice another frame than the true one? | `gray/alarm.py` | `alarm.sbatch` | `runs/<name>/alarm*.json` |
| How far generated head paths go at each sampling setting | `gray/travel.py` | `once.sbatch` | `runs/<name>/travel.json` |

## HumanML3D

| Step | Code | Job | Output in `/ptmp/enazaren/gray` |
|---|---|---|---|
| HumanML3D in the form of our samples; text features of its captions; the loader (`HmlMoments`) | `gray/hml.py` | | `hml3d/hml.npz`, `hml3d/text_tokens.npy` |
| Its motions placed into recorded frames, kept when they fit the home (`pairs`); the 3D map per home (`solid`); its joints (`joints`); some kept pairs as bodies for videos (`show`); the loader (`Placed`, `--placed` in head.py and motion.py) | `gray/place.py` | `once.sbatch` | `hml3d/placed.npz`, `ground/solid/`, `hml3d/joints.npy` |

## Comparisons with published models

| Step | Code | Job | Output |
|---|---|---|---|
| UniEgoMotion's data (Ego-Exo4D) in our form; `--ee4d` in head.py and motion.py trains on it | `gray/ee4d.py` | | `ee4d/` |
| Our chain against UniEgoMotion's saved predictions, its three tasks (`--tasks gen fore recon`); `--videos` writes bodies for videos | `gray/versus_uem.py` | `once.sbatch` | `runs/ee-motion-v1/versus_uem*.json` |
| Our chain against the EgoPriMo reimplementation (STEER) on held-out moments | `gray/versus_egoprimo.py` | `once.sbatch` | `runs/<body run>/versus_egoprimo.json` |
| Videos: ours, EgoPriMo and recorded in the Loc_08 scene (`export` on the cluster, `render` on the workstation); placed HumanML3D motions (`render --captions`); ours against UniEgoMotion on a plain floor (`floor`) | `gray/versus_scene.py`, `gray/film_blender.py`, `gray/versus_blender.py` | | workstation `gray/motion/` |
| Side-by-side bodies of several body runs on a checkered floor | `gray/versus.py` | | workstation |

Shared helpers: `gray/frames.py` (the frame anchored at "now"), `gray/census.py` (which
sentences mention walking or name a destination).

## Earlier setup: 8 s windows

Kept because its runs are the baselines. Samples are the 8 s windows of STEER's feature cache
(`/ptmp/enazaren/nymeria/egoprimo_cache_all`); the text is every sentence overlapping the window.

| What | Code | Job |
|---|---|---|
| Experiment 1: image features from the head path | `gray/features.py`, `gray/evaluate.py`, `gray/trace.py` | `features.sbatch`, `analyse.sbatch` |
| Experiment 2: joint model of head path, features and motion; also the head-only runs head-v1 and head-v2 | `gray/joint.py` | `joint.sbatch` |
| Several head paths per window, effect of the text | `gray/paths.py` | `paths.sbatch` |
| Experiment 3: replanning along recordings | `gray/loop.py` | `loop.sbatch` |
| Is the joint model's own uncertainty usable for the comparator? | `gray/spread.py` | `spread.sbatch` |
| Navigation subset of windows | `gray/census.py` | `census.sbatch` |
| Scoring against maps and text | `gray/navigate.py` | `navigate.sbatch` |

## Looking at the data

| What | Code | Job |
|---|---|---|
| Fisheye frames next to pinhole cut-outs; robot camera views | `gray/rectify.py`, `gray/views.py` | `rectify.sbatch`, `views.sbatch` |
| Where recorded paths cross annotated walls, with video frames | `gray/crossings.py` | `crossings.sbatch` |
| Every location with all recorded paths | `gray/atlas.py` | `atlas.sbatch` |
| Can obstacle distances be read off the image features? | `gray/probe.py` | `probe.sbatch` |
| The room maps against the points Aria measured | `gray/cloud.py` | `cloud.sbatch` |
| Which description of the image separates a changed scene best | `gray/spaces.py` | `spaces.sbatch` |
| Pictures: the camera view, the sentence and the generated head paths on the room map | `gray/show.py` | `show.sbatch` |
| More homes from another dataset (Ropedia Xperience 2): does reading a room improve? | `gray/ropedia.py` | `ropedia.sbatch` |

## On a workstation: prompting the chain in Loc_08

The models are small enough for one consumer GPU. With a folder that holds `env/` (Python with PyTorch and CLIP),
`data/` and `steer/` (what goes in them is described at the top of `gray/scene_worker.py`), and the Loc_08 viewer
page and scene of the earlier project:

    python3 gray/scene_server.py --static ~/cursor-workspace/loc08 --local ~/agent-workspace/gray/local

then open http://127.0.0.1:8771/viewer/egoprimo.html. Without `--local` the worker runs in a GPU session on Viper.

## Tests

Run on a workstation without the data; each makes up a tiny world in which the right answer is known.

    python -m pytest tests/test_frames.py
    python tests/smoke_features.py
    python tests/smoke_joint.py
    python tests/smoke_head.py
    python tests/smoke_motion.py

## Runs

All in `/ptmp/enazaren/gray/runs/` on Viper. A training resumes from `runs/<name>/last.pt`, so a
new training needs a name that is not taken.

| Run | Setup | Trained on | Note |
|---|---|---|---|
| `features-v1` | windows | all windows, fisheye features | Experiment 1 |
| `joint-v1` | windows | all windows, fisheye features | Experiment 2, baseline for the head path |
| `head-v1` | windows | navigation subset, pinhole features | head path only |
| `head-v2` | windows | as head-v1, plus the image grid | |
| `head-v3` | moments | navigation subset of moments, pinhole features | |
| `head-v4` | moments | plus the image grid of the current frame | |
| `head-v5` | moments | plus free space as a training target | **the planner** (4 steps, half the noise) |
| `head-v6` | moments | plus estimated depth as input | did not help |
| `head-v7` | moments | as head-v5, plus 30 % placed HumanML3D motions | 25 Nov |
| `motion-v1` | moments | body along the head path | **the body model** (4 steps, noise 0.3) |
| `motion-v2`, `motion-v3` | moments | plus HumanML3D without images (head path masked, given) | no gain |
| `motion-v4` | moments | as motion-v1, plus 30 % placed HumanML3D motions | 25 Nov |
| `ee-head-v1`, `ee-motion-v1` | UniEgoMotion's clips | its training data, no text | for the UniEgoMotion comparison |
| `compare-v1` to `compare-v4` | moments | the comparator | parked; v4 stopped at 2,500 steps |

The image grids exist twice: `patches/` holds them for the "now" frames of the window subset
(what head-v2 was trained on), `grids/` for every frame of every recording (`patches.py --all`).
Both have the same layout, so either folder can be given to `--patches`.
