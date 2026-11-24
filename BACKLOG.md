# Backlog

What is being worked on and what waits. Newest decisions first. Results go to `DESIGN.md`; this
file only says what is open.

## Focus (set by the user, 24 Nov 2026)

Before the comparator is taken further: (1) good motion quality, by also training on HumanML3D,
and (2) a comparison with EgoPriMo and UniEgoMotion trained on the same data.

## Now

- [x] **HumanML3D in our format** (`gray/hml.py`, 24 Nov). 26,682 motions (half of them mirrored copies), 54.9 h,
      encoded with the motion autoencoder; a camera pose per frame from the head joint and the average
      camera-on-head transform of Nymeria (12.3 cm; single frames differ by 4.4 cm and 18 degrees, median).
      On the workstation in `gray/local/data/hml3d/`; the archives are also on Viper in `/ptmp/enazaren/hml3d`.
- [ ] **A HumanML3D subset.** Counts are in (`gray/hml.py census`); the rule is to be chosen with the user.
- [x] Text features of the HumanML3D captions: 57,052 captions of 19,848 motions (`hml3d/text_tokens.npy`, 1.5 GB).
- [x] HumanML3D samples in the loader (`gray/hml.py: HmlMoments`): 27,192 samples of 7,498 motions of the training
      split; the body model takes a "no image" token for them and, by choice, a "no head path" token.
- [x] Two foot measures that need no recorded motion (`gray/chain.py: feet`): sliding while a toe is down, and how
      high the lower toe is above the clip's lowest point.

## Next

- [ ] **Mixed training of the body model.** 30 % of the samples from HumanML3D. `motion-v2`: their head path masked;
      `motion-v3`: their head path given. Compare with `motion-v1` on the held-out homes, with the checks of
      `gray/chain.py` including the foot measures. Not yet: a variant with the head path roughened; the head-path model
      (not trained on HumanML3D for now, since its made-up head paths would be targets there).
- [ ] **Rejection sampling into real scenes.** Place a HumanML3D motion at a recorded starting frame
      in a Nymeria home; keep it if its path goes through nothing on that home's map. Train with
      these as extra samples that have a real image.
- [ ] **EgoPriMo as a baseline** (user, 24 Nov: compare on the first 5 s). The earlier reimplementation trained on all
      of Nymeria exists (`/ptmp/enazaren/nymeria/ego_runs/egoprimo/all-medium-v1`, 500,000 steps, same held-out homes;
      generation: joints 70.8 cm on 8 s clips). Score both on the same held-out starting frames, from one given frame and
      the text, cut to 5 s. Open: it was trained with all sentences that overlap an 8 s window, ours with one sentence
      per 5 s; how to make that fair (same text for both, or retrain it on moments).
- [ ] **UniEgoMotion as a baseline** (user, 24 Nov: take it as released, and train a version of ours on the data it
      uses). On the MPI cluster under `/scratch/inf0/user/enazaren`: its code, the released weights
      (`UniEgoMotion/exp/uem_v4b_dinov2/last.ckpt`), and its processed data (`ee4d/uniegomotion`, 36 GB: motion for
      training and validation, DINOv2 image features per frame, the validation truth for its evaluation). Being copied
      to the workstation (`gray/local/uniegomotion`, `gray/local/data/ee4d`). Clips are 8 s at 10 frames per second;
      there are no captions, so this comparison is without text. Steps:
  - [ ] read its data format; bring it into our form (body in our encoding at 15 per second, camera path, its image features)
  - [ ] run its released model and its evaluation on its validation set (needs its environment and the SMPL-X model)
  - [ ] train our chain on its training set; score both on the first 5 s with its measures and ours
  - note: its repository has a two-stage configuration (a trajectory model, then motion), the same order as our chain
- [ ] **Motion quality measures** that do not need a recorded motion to compare with: foot sliding,
      and whether the feet are on the floor.

## Decided

- The comparison is our chain (head path, then body) against models that generate the body directly.
- Clip length: everyone is scored on the first 5 s.
- UniEgoMotion is used as released; ours is trained on its data.
- HumanML3D subset: 5.1 s or longer and the pelvis gets 1 m or further (33,824 samples), unless the user says otherwise.

## Running

- `motion-v2` (jobs 13656547 to 13656550, checks 13656551) and `motion-v3` (13656552 to 13656555, checks 13656556) on Viper.
- The checks with the foot measures for `motion-v1` (13656546).

## Parked

- [ ] Comparator: `compare-v4` (trained with revisits as targets) stopped at 2,500 of 20,000 steps when the comparator was
      parked (24 Nov; state in `runs/compare-v4`, resumes with `jobs/chain_train.sh`). At 2,500 steps an hour it needs 8 h.
- [ ] Comparator: the moved-furniture test in Loc_08; connecting it to the chain; what to do after a surprise.
- [ ] A scene that really changes (edited real frames, a renderer, or a robot).
- [ ] Choosing among several drawn head paths.
- [ ] How much of the "goes through something" rate is staircases, which the flat maps mark as blocked.
- [ ] More homes for reading unseen rooms: repeat the Ropedia test when the dataset is complete.
- [ ] The camera of the target robot.
