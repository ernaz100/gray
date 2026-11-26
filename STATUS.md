# Gray: where the project stands

As of 25 November 2026, evening. The short version of the results. `HANDOFF.md` says how to work
on the project; `DESIGN.md` has every experiment with its reasoning, in order; `BACKLOG.md` what
is open.

## 1. Goal

A system that takes what a head camera sees and a sentence, generates motion that works in the
scene, and notices when the world differs from what it expected (after Jeffrey Gray's
comparator): predict what the camera will show, compare, stop and replan on a mismatch. The aim
is a closed loop that can be put into a scene and prompted around.

The chain: image and text give the **head path**; the head path gives the **body**. The
**comparator** checks the head path against what the camera sees. Order of work since 24 Nov:
motion quality and comparisons with published models first, the comparator after.

## 2. Rules set by the owner

- **Robot inputs only:** camera image, text, its own head pose. No gaze, no body history at test
  time, no known map. Maps may be used to train and to score.
- **Indoor navigation** in homes.
- **5 s**, and everything judged on homes never seen in training.

## 3. The head path (`gray/head.py`)

A flow model that predicts the clean path from the past head poses, image features, the image
grid of the current frame and the sentence; in training it also predicts how far the way is
free around the camera. Scored on 3,000 held-out moments (`gray/score.py`): does the path go when
the sentence says so, does it reach a named object, does it go through something on the home's
map (`gray/ground.py`: walls, plus what Aria measured in the same place in most recordings).

| Run and sampling | Goes through something | Reaches the named object |
|---|---:|---:|
| `head-v3` (moments) | 40.8 % | 53.7 % |
| `head-v4` (+ image grid) | 36.1 % | 50.4 % |
| `head-v5` (+ free space as a training target) | 33.7 % | 58.0 % |
| `head-v5`, 4 steps | 28.5 % | 61.6 % |
| **`head-v5`, 4 steps, half the starting noise (the planner)** | **17.5 %** | **61.0 %** |
| `head-v8`: V-JEPA 2 clips in place of all DINOv3 inputs, planner | 17.0 % | 51.7 % |
| Recorded paths | 7.3 % | |

- It goes when told in 85.5 % of the cases (people 92.0 %), reaches a named object in 61 %
  against 25 % with the wrong sentence, and in the first second goes through something as rarely
  as people do.
- Less noise removes detours more than distance (`gray/travel.py`): the path gets a fifth shorter,
  its end moves 9 %. Generated paths are a quarter shorter than recorded ones at the planner setting.
- It does not read unseen rooms well (0.55 m against 0.28 m on training homes); most likely too
  few homes (39). Depth, finer grids, earlier frames and another dataset (Ropedia) did not change that.
- V-JEPA 2 in place of DINOv3 (`head-v8`, 26 Nov): obstacles level, the unseen-room free distance
  slightly better (0.52 m), but the named object reached less often (52 % against 61 %, 58 moments).
  head-v5 stays the planner; both encoders together is the next thing to try.

## 4. The body (`gray/motion.py`)

Given the head path, generates the body (pose latents of STEER's motion codec and the pelvis
path). `motion-v1`, 4 steps, noise 0.3, on held-out homes:

- Pelvis 4.7 cm from the recorded one (6.0 cm without its own past; straight under the camera: 22 cm).
- Along generated head paths the head joint stays 13 cm from the camera, as for recorded bodies.
- Toes slide 30 cm/s while down, recorded 26 cm/s.

**HumanML3D for motion quality** (`gray/hml.py`, `gray/place.py`):

- Mixed in without images (`motion-v2`, `motion-v3`): no gain.
- Placed into real recorded frames by rejection sampling: a HumanML3D motion's camera is put on
  the recorded camera, and the pair is kept if over 5 s the path stays on walked floor, no body
  part is inside a wall, and legs, trunk and head are clear of what stands in the room (a 3D map
  from the measured points); hands may touch things. 6.6 % of 612,416 draws pass, 36,766 pairs.
  Recorded bodies pass the same rule in 88 %, so it errs on the strict side.
- Trained with these pairs (`head-v7`, `motion-v4`): no gain. Collisions 17.8 against 17.5 %,
  goes when told 80.0 against 82.0 %, named object 54 against 61 % (58 moments), body unchanged.
  HumanML3D has now been tried three ways without a gain on our measures; `head-v5` and
  `motion-v1` stay the models.

## 5. Against published models

**UniEgoMotion** (released model, on its own Ego-Exo4D validation data, no text). Ours is the
chain trained on its training data (`ee-head-v1`, `ee-motion-v1`). `gray/versus_uem.py`, 3,347 to
3,678 clips, 22 joints, 5 s:

| Task | Given | UniEgoMotion | Ours | Pelvis travels: recorded / it / ours |
|---|---|---:|---:|---|
| Generation | the first frame | 33.9 cm | 32.3 cm | 0.76 / 0.59 / 0.72 m |
| Forecasting | 2 s of camera path and images | 30.7 cm | 28.4 cm | 0.75 / 0.41 / 0.73 m |
| Reconstruction | the whole camera path | 9.6 cm | 8.7 cm | fixed by the path |

- Level or slightly ahead in all three; after aligning the pose the two are within 0.3 cm.
- Standing still scores 22 cm in generation and forecasting, better than both: on this data joint
  distance rewards not moving, so travel is reported next to it.
- One saved sample of UniEgoMotion per task; ours varies by 0.3 cm over three seeds.
- Videos on a plain floor: its body often stays where it starts; ours walks, often the wrong way.

**EgoPriMo** (our reimplementation; no code was released), 1,500 held-out Nymeria moments, both
given the frame at now and one sentence (`gray/versus_egoprimo.py`):

| | Joints | Pelvis travels | Goes through something |
|---|---:|---:|---:|
| Recorded | | 1.97 m | 3.9 % |
| Standing still | 83.9 cm | 0 | |
| EgoPriMo reimplementation | 76.5 cm | 1.49 m | 27.1 % |
| Ours, from one frame | 52.3 cm | 1.50 m | 12.6 % |
| Ours, from one frame, full noise | 61.9 cm | 2.11 m | 26.1 % |

Part of the gap is sampling (ours with reduced noise, it without), part is that the test samples
are our kind. The owner decided not to pursue it further.

## 6. The comparator (parked)

Predicts the image feature (`compare-v2`) or the 7 x 7 grid (`compare-v3`) along the head path
with its uncertainty. Calibrated, and turning is explained away. Alarm rate one second ahead:
5 % on the true frame (by construction), 13 to 16 % on the same unchanged place seen again, 50 %
on the same place another day, 91 % on a different scene. Too many false alarms when nothing
changed. `compare-v4` (revisits as targets) stopped at 2,500 of 20,000 steps.

## 7. Open

See `BACKLOG.md` and section 8 of `HANDOFF.md`.
