# Gray: where the project stands

As of 23 November 2026. This file is the short version. `DESIGN.md` has every experiment with
its reasoning and numbers in the order they happened; `README.md` says where code and data are.

## 1. Goal

A system that takes what a head camera sees and a text prompt, generates motion that works in
the scene, and notices when the world differs from what it expected (after Jeffrey Gray's
comparator): predict what the camera will show, compare with what it shows, stop and replan on
a mismatch. The aim is a closed loop that can be put into a scene and prompted around.

Order of the parts: text and image give the head path; the head path gives what the camera
should show along it (the comparator); both give the body motion.

## 2. Rules set by the user

- **Robot inputs only.** The model gets nothing a robot would not have: camera image, text, its
  own pose. No gaze, no human body history, no known map of the scene.
- **Indoor navigation.** No standing around, no outdoor walks.
- Room maps and other truths may be used to train and to score, never as input.

## 3. Data that was prepared

All on Viper under `/ptmp/enazaren/gray`. Source: Nymeria, 1,074 recordings, 17.7 M frames at 15 per second.

| What | Folder | Note |
|---|---|---|
| Pinhole view, one vector per frame | `pinhole/` | the fisheye frame cut to a normal camera of 79.6 degrees; DINOv3 class token |
| Image grid 7 x 7 per frame | `grids/` | the patch features of the same pass, 621 GB |
| Estimated depth per frame | `depth/` | Depth Anything V2; turned out not to help |
| Head path of each recording | `tracks/` | one continuous track, matches the old cache to 0.22 cm |
| Moments | `moments/` | see below |
| Room maps from annotated boxes | `layout/` | walls reliable, furniture boxes too rough |
| Free and blocked space per home | `ground/` | the truth used now, see section 5 |

**Moments.** Nymeria's narration describes each recording in pieces of 5 s, one sentence each. A
moment is one piece: "now" is where the sentence begins, the model gets the 2 s before and
predicts the 5 s the sentence describes. 167,796 moments; the navigation subset (indoor, mostly
walking) has 44,754, of which 39,130 for training and 5,624 held out in homes never seen in training.

## 4. Part one: the head path

**Model** (`gray/head.py`): a flow model that predicts the clean path. Input: past head poses
and image vectors, the sentence, the image grid of the current frame. Output: the head poses of
the next 5 s. In training it also has to say how far the way is free around the camera.

**Runs, all scored on the same 3,000 held-out moments** (moments that go 1 m or further):

| Run | What it adds | Goes through something | Reaches the named object |
|---|---|---:|---:|
| `head-v3` | moments | 40.8 % | 53.7 % |
| `head-v4` | image grid of the current frame | 36.1 % | 50.4 % |
| `head-v5` | free space as a training target | 33.7 % | 58.0 % |
| `head-v6` | estimated depth as input | 34.1 % | 57.3 % |
| `head-v5`, sampled in 4 steps | fewer sampling steps, no retraining | 28.5 % | 61.6 % |
| `head-v5`, sampled in 1 step | | 26.0 % | 62.5 % |
| `head-v5`, 4 steps, half the starting noise | less noise, no retraining | 17.5 % | 61.0 % |
| `head-v5`, 4 steps, no noise (one path per moment) | | 13.9 % | 62.1 % |

Recorded paths: 7.3 %. Recorded paths turned by 90 degrees: 71.3 %.

**What works**

- It follows the text: it goes when the sentence mentions walking in 85.5 % of the cases
  (people: 92.0 %), and in 12.3 % when it does not (people: 15.9 %).
- It reaches a named object in about 60 % of the draws, against 25 % with the wrong sentence.
- In the first second, which a system that replans executes, it goes through something about
  as rarely as recorded paths do (0.9 to 2.1 % against 1.1 %).

**What does not**

- Over the full 5 s one path in six to seven still goes through something (recorded: one in fourteen).
- It cannot read the layout of a home it has not seen better than to about 0.55 m. On homes it
  trained on it reaches 0.28 m: it learns rooms, not how to read a room. Most likely cause: only
  39 training homes. No feature changed this (finer grids, depth, earlier frames).

**Things tried that did not help:** text guidance at sampling; estimated depth; explorative
training (best of K noises, arXiv 2607.27372).

**What helped without retraining:** fewer sampling steps and less starting noise. The path the
model produces without noise is its best single guess; noise spreads the draws mostly into worse ones.

**Decision (22 Nov, setting updated 23 Nov):** `head-v5` sampled in 4 steps with half the starting noise is the planner. The head path is good enough
to move on. Open side task: choosing among several draws. Among 8 draws one is nearly always
clear (3.7 %); picking by the model's own free-space estimate gives about 20 %.

## 5. How paths are judged

Not by distance to the recorded path, since many paths are valid for one sentence
(`gray/score.py`):

- **Goes when told:** gets 1 m or further, against whether the sentence mentions walking.
- **Destination:** ends within 1 m of an object the sentence names.
- **Goes through something:** a point in a blocked cell while moving faster than 0.5 m/s.

The truth for the last one (`gray/ground.py`) took several rounds. The annotated furniture
boxes are in the right place but include space people walk through. Now, per home and from all
its recordings: **free** is where somebody walked upright; **blocked** is walls plus cells where
Aria measured points in at least 30 % of the recordings; the rest is unknown.

## 6. Part two: the comparator (in progress)

**Model** (`gray/compare.py`): given the past, the sentence, the current image grid and the head
poses of the next 5 s, predict what the camera will show at each of those frames, and how
uncertain that is. One pass, no sampling.

**Signal:** the "surprise", the error divided by the predicted uncertainty. It should be about
1 when the world is as expected, whatever the head did.

**`compare-v2`** (predicts the one vector per frame; trained on all moments), on unseen homes:

- Mismatch 0.529, against 1.069 for copying the last frame.
- Calibrated: surprise 1.10.
- Turning is explained away: the raw mismatch rises from 0.34 to 0.75 with turning, the
  surprise stays at 1.1.

**Does it notice a changed world?** (`gray/alarm.py`.) Recordings have no unexpected changes,
so the comparator is handed another frame than the true one. Alarm rate 1 s ahead, with a noise
floor added to the uncertainty:

| Observed frame | Alarm rate | Should be |
|---|---:|---|
| The true frame | 5 % | 5 % by construction |
| The same place seen again later, nothing changed | 16 % | near 5 % |
| The same place on another day | 50 % | clearly raised |
| A different scene | 79 % | high |

- The order is right and a different scene is caught four times in five.
- Too many false alarms when nothing changed, and "another day" is not well separated from
  that. Cause: the one vector per frame describes the whole image, including hands, people and
  objects being handled.

Averaging the alarm over five frames does not reduce the false alarms.

**`compare-v3`** predicts the 7 x 7 image grid instead of the one vector. It predicts as well
(mismatch 0.526 against 1.156 for copying, surprise 1.08, no alarm on turning) and catches a
different scene more often: 91 % against 79 %. It does not lower the false alarms when nothing
changed (16 %; 13 % when only the cells on the path are compared). So those false alarms do not
come from the edges of the image. They are a matter of the first second, where the model is most
sure and two views of an unchanged place differ by more than it allows for.

**Next for the comparator:** train it with revisits as targets, so that its own uncertainty
learns how much an unchanged place varies; and a test in Loc_08, where it is known which
furniture stood where, to see whether a moved piece is noticed.

## 7. Not started

- **Body motion** given the head path. An earlier experiment on the old setup put the pelvis
  7 cm from the recorded one when the head path was given, so it is expected to be the easier part.
- **The closed loop** itself: plan, move, compare, replan.
- **A test with a truly changed scene.** Needs a renderer or a robot. The rendered Loc_08 scene
  gives image features that differ strongly from real ones (cosine 0.45), which is unsolved.

## 8. In progress (23 Nov, afternoon): three things at once

1. **Comparator.** `compare-v4` trains with revisits as targets (the same place seen again later
   in the recording; 48 % of all frames have one), so that its uncertainty covers how an unchanged
   place varies. Waiting in the queue (job 13603988); the alarm test follows. Then the test in
   Loc_08 with known furniture positions.
2. **Body motion.** The body of every recording is now one continuous track (`gray/body.py`,
   encoded again from the motion capture). The body model (`gray/motion.py`, `motion-v1`, job
   13605449) gets the head path of the next 5 s and generates the body along it. Waiting in the queue.
3. **More homes** (Ropedia Xperience 2, `gray/ropedia.py`). Tested and not conclusive: of 105
   indoor walk-throughs online only 41 have a usable mesh. Adding their 27 training scenes does
   not improve reading held-out Nymeria homes (error 0.81 m against 0.82 m), and the two datasets
   transfer poorly to each other. Too few and too different to say whether hundreds of homes
   would help. Worth repeating when the rest of the dataset and its registration are published.

## 9. Open decisions

1. If the grid comparator does not cut the false alarms: what the comparator should compare.
2. Whether to build a chooser among several drawn paths now or leave it.
3. When to start on the body motion.
4. The camera of the target robot. The Unitree G1's camera is, by its public description,
   mounted 47.6 degrees below level on a head without a neck; set aside, not verified on a robot.

## 10. Things worth knowing before changing anything

- Windows of 8 s (`gray/joint.py`, runs `joint-v1`, `head-v1`, `head-v2`) are the earlier
  setup, kept as baselines. New work uses moments.
- Both models memorise the training homes. Keep the state that is best on held-out moments
  (`compare.py` does), and judge on held-out homes only.
- Opening the files of one recording takes about half a second on Viper's file system; data
  loaders are slow for the first pass over the recordings.
- Viper's 15-minute test partition starts jobs within minutes; the main queue can take hours.
- Nothing in the repository is committed.
