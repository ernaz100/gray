# Handoff

For whoever takes this project over. As of 25 November 2026, evening. Read this first; it points
to everything else.

| File | What it is for |
|---|---|
| `HANDOFF.md` | this file: how to work on the project, where things are, what is open |
| `STATUS.md` | the results so far, short |
| `BACKLOG.md` | open tasks and decisions, kept current |
| `DESIGN.md` | every experiment with its reasoning and numbers, in the order they happened (long; search it) |
| `README.md` | which script does what, with its job file and output folder |
| `docs/index.html` | the same as a page with pictures and videos; open it in a browser after cloning. The **Status** tab is the overview |
| `jobs/clusters/README.md` | running on a cluster |

## 1. The project in five sentences

From what a head-worn camera sees and a sentence, generate the motion of the person (or a
robot) for the next 5 s, in a way that works in the room. It is built as a chain: a **head-path
model** plans where the camera goes, then a **body model** generates the body along that path.
A third part, the **comparator** (after Jeffrey Gray), predicts what the camera will see along
the plan and raises an alarm when the world differs; it is parked. The goal is a closed loop
that can be put into a scene and prompted around. Data: Nymeria (egocentric recordings in 49
homes with narration and motion capture).

## 2. Where things stand

**Works:**

- Head path (`head-v5`): follows the text and avoids walls and furniture far better than chance,
  but one path in six still goes through something over 5 s (recorded paths: one in fourteen).
- Body along a given head path (`motion-v1`): 4.7 cm pelvis error on held-out homes.
- Against the two baselines: level with UniEgoMotion (released model) on its own data in all
  three of its tasks; clearly ahead of our reimplementation of EgoPriMo on Nymeria. Numbers in
  `STATUS.md`.

**Tried and did not help:** HumanML3D, three ways. Last: its motions placed into real recorded
frames by rejection sampling with a whole-body collision check, and the two models retrained
with them (`head-v7`, `motion-v4`): no gain on any of our measures.

**Parked:** the comparator (`compare-v4` stopped at 2,500 of 20,000 steps).

**Current focus, set by the project owner on 24 Nov:** motion quality (HumanML3D) and the
comparison with UniEgoMotion and EgoPriMo come before the comparator. The comparison with
UniEgoMotion matters most, since it is a released model.

## 3. Rules the owner has set

These hold for any change to the project.

- **Robot inputs only.** Models get what a robot would have: camera image, text, its own head
  pose. No gaze, no human body history as input at test time, no known map of the room. Maps
  and other truths may be used to train and to score, never as input.
- **Indoor navigation** is the task: walking in homes, not standing or outdoor walks.
- **Every model is scored on the first 5 s.** Comparisons are our chain (head path, then body)
  against models that generate the body directly.
- **Judge on held-out homes only.** The models memorise training homes.
- **Never judge generation by distance to the recorded body alone.** Standing still beats every
  model on that measure on UniEgoMotion's data. Every table carries a standing-still row and how
  far the bodies travel.

## 4. How to work (environment and conventions)

**Clusters.** Viper (MPCDF, AMD MI300A) holds all data and runs everything. Raven is meant to
mirror it but is not set up yet. MPI is not used. Access is through the owner's `hpc` helper on
the workstation (`hpc run viper <command>`); multi-factor login is done by the owner
(`hpc connect viper`), never by an agent.

| | |
|---|---|
| Repository on Viper | `/u/enazaren/projects/gray` (pulls from GitHub `ernaz100/gray`) |
| Data and runs on Viper | `/ptmp/enazaren/gray` (`runs/<name>/` per training or evaluation) |
| Python on Viper | `/u/enazaren/envs/torch/bin/python` (ROCm; needs `TORCH_BLAS_PREFER_HIPBLASLT=0`) |
| STEER repository (codec, EgoPriMo reimplementation, SMPL) | `/u/enazaren/projects/STEER` |
| Repository on the workstation (the working copy) | `/home/ericnazarenus/agent-workspace/gray/repo` |
| Workstation data and environments | `/home/ericnazarenus/agent-workspace/gray/local/` |
| Rendered videos on the workstation | `/home/ericnazarenus/agent-workspace/gray/motion/` |

**Running a training:** `jobs/submit.sh viper <hours> <module> <run name> [arguments]` from the
workstation. It submits a chain of 1-hour jobs on Viper's `apu1` partition that resume one
another (`jobs/run.sbatch`); `@data` in the arguments stands for the data folder. 1-hour jobs
start much sooner than long ones. Example, as `head-v7` was started:

    jobs/submit.sh viper 5 head head-v7 --patches @data/grids --ground @data/ground --placed @data/hml3d

**Running an evaluation or a one-off script:** `jobs/once.sbatch` on the 15-minute test
partition, from a Viper login shell:

    cd /u/enazaren/projects/gray && git pull && source jobs/clusters/viper.sh
    sbatch -p apudev -t 15 $GRAY_SBATCH --export=ALL,GRAY_CLUSTER=viper -D $GRAY_CODE \
        -o $GRAY_DATA/runs/gray-eval-%j.log jobs/once.sbatch versus_uem --ee4d @data/ee4d --uem @data/uem \
        --head-run @data/runs/ee-head-v1 --body-run @data/runs/ee-motion-v1

The test partition starts jobs within minutes but makes a user wait longer after many
submissions in a row. Never compute on login nodes.

**Workstation.** RTX 3090, 125 GB memory, shared with the owner's own work. Rendering (Blender)
runs here, since only the workstation has Blender and the Loc_08 scene file. Agree with the owner
before running anything heavy there; no training during the day; one process in the foreground
at a time. Two trainings at once crashed it on 25 Nov.

**Approvals.** Submitting or cancelling jobs, and changing or deleting files on a cluster, need
the owner's approval, with two standing exceptions: trainings of this project on Viper and their
evaluations may be submitted, and result updates may be committed and pushed to `ernaz100/gray`.

**Commits.** Work in the repository on the workstation, commit often, push
(`git push git@github.com:ernaz100/gray.git main`), then `git pull` on Viper. Results go into
`DESIGN.md` (new section at the end), `BACKLOG.md`, `STATUS.md` and the Status tab of
`docs/index.html` together. A single-file copy of the page for sharing is rebuilt by embedding
`docs/img` and `docs/media` as base64 (`gray-overview-standalone.html` on the workstation).

## 5. The models and runs that matter

| Run | Part | Trained on | Use |
|---|---|---|---|
| `head-v5` | head path | Nymeria moments, image grid, free space as a training target | **the planner**, sampled in 4 steps with half the starting noise |
| `motion-v1` | body | Nymeria moments | **the body model**, 4 steps, noise 0.3 |
| `head-v7`, `motion-v4` | as above | plus 30 % HumanML3D motions placed into recorded frames | no gain (25 Nov) |
| `motion-v2`, `motion-v3` | body | plus HumanML3D without images | no gain; superseded |
| `ee-head-v1`, `ee-motion-v1` | head path, body | UniEgoMotion's data (Ego-Exo4D), no text | for the UniEgoMotion comparison |
| `head-v6` | head path | as head-v5 plus estimated depth | did not help (22 Nov) |
| `compare-v3`, `compare-v4` | comparator | | parked |

Sampling matters as much as training: fewer steps and less starting noise cut collisions of the
head path from 34 % to 17.5 %, mostly by removing detours, partly by shorter paths. Without text
(UniEgoMotion's data) low noise makes the chain stand still, so there it is sampled in full
(20 steps, noise 1). Which setting a table uses is always stated.

## 6. Data

All under `/ptmp/enazaren/gray` on Viper.

| Folder | What |
|---|---|
| `moments/moments.npz` | the samples: "now" where a narration sentence begins, 2 s before, 5 s after; 39,130 training and 5,624 held-out (homes never seen) in the navigation subset |
| `pinhole/`, `grids/` | image features per frame (DINOv3, the fisheye frame cut to a normal camera), and a 7 x 7 grid |
| `tracks/`, `body/` | head path and body of each recording as one track |
| `ground/<home>.npz` | where people walked and what is blocked, per home; the truth for "goes through something" |
| `ground/solid/<home>.npz` | the same in 3D (10 cm cubes), for the whole-body check |
| `hml3d/` | HumanML3D in our form; `placed.npz` the accepted placements; `joints.npy` its joints |
| `ee4d/` | UniEgoMotion's data in our form |
| `uem/` | UniEgoMotion's saved predictions for its three tasks, its validation truth, SMPL-X |

UniEgoMotion's own code, weights and environment are on the workstation
(`local/uniegomotion`, `local/env-uem`), not on Viper. Its predictions were made there once
(one sample per clip and task).

## 7. What is in flight

Nothing runs on the clusters or the workstation.

- **Ego-Exo4D access** has been requested by the owner and is pending. With it, UniEgoMotion and
  our chain can be shown in the real rooms (video and point cloud per recording) instead of on a
  plain floor.

## 8. Open decisions and next steps

In the order I would take them:

1. **HumanML3D: stop or change approach** (owner's decision). Untried: placed motions with their
   own 2 s of history instead of single-frame starts; a measure of motion variety, which is what
   HumanML3D should add and which none of our measures looks at; pretraining on HumanML3D, then
   Nymeria.
2. **Strengthen the UniEgoMotion comparison** (the owner's priority): draw several samples from
   it (needs its environment on Viper, or a night on the workstation; about 1.5 h per run), and
   add measures that do not reward caution, compared as distributions over clips. Its semantic
   measures need the TMR model, which we do not have.
3. **UniEgoMotion in its rooms** once Ego-Exo4D access is there.
4. **Raven** in sync with Viper when it is back up (`jobs/clusters/README.md`).
5. **The comparator**, after the comparisons: `compare-v4` resumes from its state; then the
   moved-furniture test in Loc_08 and connecting it to the chain.

Owner's decisions already taken (do not reopen): EgoPriMo with our sampling is not worth doing;
UniEgoMotion is used as released and ours is trained on its data; clips are cut to 5 s.

## 9. Things that will trip you up

- **Run names are reused by mistake easily.** Trainings resume from `runs/<name>/last.pt`; a new
  training under an existing name fails or, worse, continues the old run. Check `ls runs/` first.
  (`head-v6` was nearly overwritten this way; the run on the placed pairs is `head-v7`.)
- **The Loc_08 mesh scene is rougher than the maps.** Its furniture comes from annotated boxes
  that include space people walk through. Bodies that cross furniture in the videos are often
  fine on the map; judge collisions by the map.
- **Collision rules are stricter than reality.** Recorded paths pass "not blocked" only in 80 % of
  samples; recorded bodies pass the whole-body rule in 88 %. State that next to any rate.
- **Loc_08 is a training home** of both our models and EgoPriMo. Videos there show how motions
  look, not how well they generalise.
- **UniEgoMotion's data has no text and no scene geometry**, and its clips are 10 frames per
  second (ours 15). `gray/versus_uem.py` handles the frames and the coordinate change; its own
  8 s number reproduces within 1.4 cm, which is the check that the conversion is right.
- **Opening a recording's files on Viper takes about half a second**; first passes of data
  loaders are slow.
- **The workstation folder `gray/STATUS.md` outside the repository is an old copy**; the
  repository is the source.
