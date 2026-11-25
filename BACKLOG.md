# Backlog

What is being worked on and what waits. Newest decisions first. Results go to `DESIGN.md`; this
file only says what is open.

## Focus (set by the user, 24 Nov 2026; updated 25 Nov)

Before the comparator is taken further: (1) good motion quality, by also training on HumanML3D,
and (2) a comparison with EgoPriMo and UniEgoMotion trained on the same data. Of the two
comparisons, UniEgoMotion matters most (a released model); EgoPriMo is not taken further.

## Now

- [x] **HumanML3D in our format** (`gray/hml.py`, 24 Nov). 26,682 motions (half of them mirrored copies), 54.9 h,
      encoded with the motion autoencoder; a camera pose per frame from the head joint and the average
      camera-on-head transform of Nymeria (12.3 cm; single frames differ by 4.4 cm and 18 degrees, median).
      On the workstation in `gray/local/data/hml3d/`; the archives are also on Viper in `/ptmp/enazaren/hml3d`.
- [x] **A HumanML3D subset**: 5.1 s or longer, pelvis 1 m or further from its start, training split (see Decided).
- [x] Text features of the HumanML3D captions: 57,052 captions of 19,848 motions (`hml3d/text_tokens.npy`, 1.5 GB).
- [x] HumanML3D samples in the loader (`gray/hml.py: HmlMoments`): 27,192 samples of 7,498 motions of the training
      split; the body model takes a "no image" token for them and, by choice, a "no head path" token.
- [x] Two foot measures that need no recorded motion (`gray/chain.py: feet`): sliding while a toe is down, and how
      high the lower toe is above the clip's lowest point.

## Next

- [x] **Mixed training of the body model** (`motion-v2` masked, `motion-v3` given; 25 Nov). No gain on held-out homes by
      our measures: pelvis 5.2 and 4.7 cm against 4.7 cm, feet slide as before (30.6 against 30.2 cm/s; recorded 25.5).
- [x] **Look at the motion** of `motion-v1` and `motion-v3` on the same moments: four videos (`gray/versus.py`, workstation
      `gray/motion/versus/`), sent to the user on 25 Nov; no comment yet.
- [ ] Tighten the foot-slide measure, which counts a toe pushing off as down.
- [ ] If HumanML3D is to be pursued: a larger share, the head path roughened, or pretraining on it and then Nymeria.
- [x] **Rejection sampling into real scenes** (`gray/place.py`, 25 Nov): 6.6 % of draws fit (path on walked floor, whole
      body clear of walls and of what stands there, camera height within 10 cm); 36,766 pairs.
- [x] **Training on the pairs**: `head-v7` and `motion-v4` trained (40,000 steps each, 25 Nov). Training logs: body unchanged,
      head path 4 to 5 cm further from the recorded path per draw.
- [x] **Scored** like head-v5 and motion-v1: no gain (collisions 17.8 against 17.5 %, named object 54 against 61 %, body
      unchanged). HumanML3D tried three ways without a gain; head-v5 and motion-v1 stay the models.
- [ ] **Decide with the user** whether to stop on HumanML3D, or try what has not been tried: placed motions with their
      own history instead of single-frame starts, a measure of motion variety, or pretraining on HumanML3D.
- [x] Videos of accepted placements in Loc_08 (`gray/place.py show`, workstation `gray/motion/placed/`, and a map picture).
- [x] UniEgoMotion against ours as video on a plain floor (six clips, workstation `gray/motion/versus-uem/`).
- [ ] UniEgoMotion in its rooms: needs Ego-Exo4D access (requested by the user, pending).
- [ ] Several samples of UniEgoMotion (its environment on Viper, or a night on the workstation), so that its spread is known.
- [x] **EgoPriMo as a baseline** (`gray/versus_egoprimo.py`, 25 Nov): on 1,500 held-out moments, at the same amount of
      movement, our chain is a third nearer to the recorded body (52.3 against 76.5 cm) and goes through something half as
      often (12.6 against 27.1 %). It is our reimplementation, used outside how it was trained.
- [ ] **UniEgoMotion as a baseline** (user, 24 Nov: take it as released, and train a version of ours on the data it
      uses). Its code, the released weights and its processed data are on the workstation (`gray/local/uniegomotion`,
      `gray/local/data/ee4d`; first fetched from the MPI cluster, which is no longer used); its saved predictions and
      validation truth also on Viper (`uem/`). Clips are 8 s at 10 frames per second; no captions, so this comparison
      is without text. Steps:
  - [x] its data in our form (`gray/ee4d.py`; on Viper in `ee4d/`, on the workstation in `gray/local/data/ee4d`)
  - [x] our head-path and body models trained on it (`ee-head-v1`, `ee-motion-v1`, Viper)
  - [x] its released model and its evaluation on its validation set, on the workstation: generation from the first image
        MPJPE 0.415 m over 8 s (0.249 m in the first 2 s), forecasting 0.271 m, reconstruction 0.109 m. To do: check against
        the paper's table; its semantic measures need a model we do not have.
  - [x] our chain against UniEgoMotion on its validation clips (`gray/versus_uem.py`, Viper): level with it or slightly ahead when
        sampled in full (joints 32.3 against 33.9 cm, travel 0.72 against 0.59 m; recorded 0.76 m). Standing still beats both
        on joint distance, so that measure alone is not used any more.
  - [x] three seeds: ours 32.3 to 32.6 cm against 33.9 cm
  - [x] its forecasting and reconstruction tasks: 28.4 against 30.7 cm, and 8.7 against 9.6 cm
  - [ ] measures that do not reward caution, as distributions over clips
  - note: its repository has a two-stage configuration (a trajectory model, then motion), the same order as our chain
- [ ] **Motion quality measures** that do not need a recorded motion to compare with: foot sliding,
      and whether the feet are on the floor.

## Decided

- EgoPriMo: not taken further (no sampling with our settings, no retraining); user, 25 Nov.
- Clusters: Viper and Raven only, kept in sync; MPI is not used (user, 25 Nov).
- Collision check for placed motions: every body part, not only head and pelvis (user, 25 Nov).

- The comparison is our chain (head path, then body) against models that generate the body directly.
- Clip length: everyone is scored on the first 5 s.
- UniEgoMotion is used as released; ours is trained on its data.
- HumanML3D subset: 5.1 s or longer and the pelvis gets 1 m or further (33,824 samples), unless the user says otherwise.

## Running

Nothing.

## Infrastructure

- [x] `jobs/submit.sh`: submit a training to a named cluster or to the one that starts it soonest (Viper set up).
- [ ] Set up Raven like Viper when it is back up (`jobs/clusters/README.md`): repository, Python, and the data. Low priority.

## Parked

- [ ] Comparator: `compare-v4` (trained with revisits as targets) stopped at 2,500 of 20,000 steps when the comparator was
      parked (24 Nov; state in `runs/compare-v4`, resumes with `jobs/chain_train.sh`). At 2,500 steps an hour it needs 8 h.
- [ ] Comparator: the moved-furniture test in Loc_08; connecting it to the chain; what to do after a surprise.
- [ ] A scene that really changes (edited real frames, a renderer, or a robot).
- [ ] Choosing among several drawn head paths.
- [x] How far the planner's paths go at each noise setting (`gray/travel.py`): less noise shortens the path by a fifth but moves
      its end by 9 %; generated paths are a quarter shorter than recorded ones at the planner's setting.
- [ ] How much of the "goes through something" rate is staircases, which the flat maps mark as blocked.
- [ ] More homes for reading unseen rooms: repeat the Ropedia test when the dataset is complete.
- [ ] The camera of the target robot.
