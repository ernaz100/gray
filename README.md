# gray

**Overview with pictures:** open [`docs/index.html`](docs/index.html) in a browser after cloning. It summarises the state of the research, with figures and the main numbers.


Egocentric image and text in, body motion out, with a comparator that notices when the world
differs from what was predicted. `DESIGN.md` has the reasoning, every experiment and its numbers
in the order they happened. This file only says where things are. `STATUS.md` is the short summary of where the project stands.

Everything runs on Viper: Python `/u/enazaren/envs/torch/bin/python`, jobs in `jobs/` (each
names its log and output at the top), outputs under `/ptmp/enazaren/gray`.

## Current work: the head path on moments

The model that predicts where the head goes, from text and what the camera saw. A sample starts
where a narration sentence starts.

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
| Workstation: videos of generated motion on the room map | `gray/animate.py` | | |
| Workstation: a video with the views the models were given, the body as a mesh in the home's scene, and the planned camera's view (Blender) | `gray/film.py`, `gray/film_blender.py`, `gray/seen.py` | | |
| For every frame a later frame from nearly the same pose (targets for the comparator) | `gray/revisits.py` | `revisits.sbatch` | `revisits/<id>.npy` |
| Does the comparator notice another frame than the true one? | `gray/alarm.py` | `alarm.sbatch` | `runs/<name>/alarm*.json` |

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

| Run | Setup | Trained on | Note |
|---|---|---|---|
| `features-v1` | windows | all windows, fisheye features | Experiment 1 |
| `joint-v1` | windows | all windows, fisheye features | Experiment 2, baseline for the head path |
| `head-v1` | windows | navigation subset, pinhole features | head path only |
| `head-v2` | windows | as head-v1, plus the image grid | |
| `head-v3` | moments | navigation subset of moments, pinhole features | |

The image grids exist twice: `patches/` holds them for the "now" frames of the window subset
(what head-v2 was trained on), `grids/` for every frame of every recording (`patches.py --all`).
Both have the same layout, so either folder can be given to `--patches`.
