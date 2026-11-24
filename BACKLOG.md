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
- [ ] **Samples from HumanML3D in the loaders**, with a "no image" token, so that `motion.py` and `head.py`
      can train on a mixture.

## Next

- [ ] **Mixed training.** Body model, and separately the head path, on Nymeria plus the subset;
      compare with the Nymeria-only runs on the held-out homes.
- [ ] **Rejection sampling into real scenes.** Place a HumanML3D motion at a recorded starting frame
      in a Nymeria home; keep it if its path goes through nothing on that home's map. Train with
      these as extra samples that have a real image.
- [ ] **EgoPriMo as a baseline** (user, 24 Nov: compare on the first 5 s). The earlier reimplementation trained on all
      of Nymeria exists (`/ptmp/enazaren/nymeria/ego_runs/egoprimo/all-medium-v1`, 500,000 steps, same held-out homes;
      generation: joints 70.8 cm on 8 s clips). Score both on the same held-out starting frames, from one given frame and
      the text, cut to 5 s. Open: it was trained with all sentences that overlap an 8 s window, ours with one sentence
      per 5 s; how to make that fair (same text for both, or retrain it on moments).
- [ ] **UniEgoMotion as a baseline** (user, 24 Nov: take it as released, and train a version of ours on the data it
      uses). Needs: its released weights and code; EE4D-Motion (on the MPI cluster under `/scratch/inf0/user/enazaren/ee4d`)
      brought into our form (body latents, head path, image features); a run of our chain on it. EE4D-Motion has
      no captions, so this comparison is without text. Not started; the MPI cluster is not connected.
- [ ] **Motion quality measures** that do not need a recorded motion to compare with: foot sliding,
      and whether the feet are on the floor.

## Decided

- The comparison is our chain (head path, then body) against models that generate the body directly.
- Clip length: everyone is scored on the first 5 s.
- UniEgoMotion is used as released; ours is trained on its data.
- HumanML3D subset: 5.1 s or longer and the pelvis gets 1 m or further (33,824 samples), unless the user says otherwise.

## Running

- [ ] `compare-v4` (comparator trained with revisits as targets): six 1-hour jobs queued on Viper
      (13641519 to 13641524) and its alarm test (13641525).

## Parked

- [ ] Comparator: the moved-furniture test in Loc_08; connecting it to the chain; what to do after a surprise.
- [ ] A scene that really changes (edited real frames, a renderer, or a robot).
- [ ] Choosing among several drawn head paths.
- [ ] How much of the "goes through something" rate is staircases, which the flat maps mark as blocked.
- [ ] More homes for reading unseen rooms: repeat the Ropedia test when the dataset is complete.
- [ ] The camera of the target robot.
