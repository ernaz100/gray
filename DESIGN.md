# Gray: a comparator for egocentric motion generation

Started 18 Nov 2026. Same problem as STEER (`/u/enazaren/projects/STEER`): from an egocentric
image and a text prompt, generate body motion that works in the scene. The new part is a closed
loop inspired by Jeffrey Gray's comparator: predict what the camera will show, compare with what
it does show while moving, and stop and replan when the two disagree.

## The model as agreed so far

Built in the style of ActionPlan (`STEER/models/ActionPlan-Code`): one token per frame, a plan that
is denoised first and the motion that is denoised given the plan.

| Part of the token | Contents | ActionPlan's counterpart |
|---|---|---|
| Head | head pose (position, 6D rotation) in the frame anchored at "now" | — |
| Feature | DINOv3 class token of the frame, raw 384-d for now | frame-level text latent (the plan) |
| Motion | pose latent and root channels, as in STEER | motion latent |

- **Order.** Image and text give the head path; head path and image give the features along it;
  both give the motion.
- **Three noise levels**, one per part and per frame (ActionPlan has two, and one level per clip
  for its plan). Past tokens are clean because they were observed; future ones are generated.
  "Head clean, feature noisy" is the case the comparator uses.
- **Comparator.** The feature is predicted for the pose the head is really at, not the planned
  one, so a mismatch means the world differs, not that the path was left. On recordings the real
  pose is put into the head part; in a deployed loop the planned path is the executed one.
- **Trigger.** A plain threshold on the mismatch at first. Any trigger replans from the current
  frame; a large one stops first. A predicted uncertainty (ActionPlan already outputs a variance)
  is added only if this fires too often.
- **Replanning.** As ActionPlan's streaming: the last tokens stay fixed at the start of the new
  window. Here that includes the observed features and the real head poses, so the history is in
  the window. Memory beyond the window is deferred.
- **Frame.** Re-anchored at every replan: origin on the floor under the camera now, +z along its
  horizontal view direction now (`gray/frames.py`). In training, "now" is a random token of the
  window and everything before it is history.

## Goal and tests

The goal is a closed loop that can be put in a scene and prompted around.

1. **On Nymeria recordings** (training and first checks): feature prediction against the recorded
   frames; whether mismatches fall on real events; whether replanning on mismatch beats replanning
   at fixed intervals for the same number of replans.
2. **In a rendered scene** (Loc_08, `STEER/egoprimo/scene_worker.py`): the generated head path
   drives the render and the render feeds the comparator.

## Open

- Rendered against real images: if their DINOv3 features differ, the comparator sees a permanent
  mismatch. To check early, at poses where a real frame exists.
- A static reconstruction only surprises with unseen parts; testing the stop needs objects
  inserted or moved in the render. Format of the Loc_08 reconstruction not yet known here.
- Whether the class token is fine enough (experiment 1). Fallback: pooled patch tokens.
- Raw 384-d features against a compressed latent for the generative model.

## Experiment 1: are the features predictable from the head path?

`gray/features.py`, job `jobs/features.sbatch`. Uses STEER's cache of all 1,074 sequences
(`/ptmp/enazaren/nymeria/egoprimo_cache_all`) as it is. A small transformer gets 1 to 30 frames
up to "now" and the head poses of the rest of the 8 s window, and predicts the hidden frames'
class tokens. It is compared on held-out locations with copying the frame at now and with the
same model without poses, by time since now, by turning and by distance.

### Result (18 Nov 2026, job 13289971, 61 min on one MI300A)

20,000 steps each, batch 256, 11 M parameters; 418,757 training windows, 4,096 of the 50,508
held-out windows (6 held-out locations). Mean squared error on standardized features, where 1.0
is predicting the average frame; one frame given. Log:
`/ptmp/enazaren/gray/runs/gray-features-13289971.log`, results in `runs/features-v1/`.

| Time since now | With poses | Time only | Copy |
|---|---:|---:|---:|
| 0 to 0.5 s | 0.284 | 0.282 | 0.268 |
| 0.5 to 1 s | 0.413 | 0.453 | 0.505 |
| 1 to 2 s | 0.493 | 0.583 | 0.681 |
| 2 to 4 s | 0.565 | 0.697 | 0.847 |
| 4 to 8 s | 0.621 | 0.771 | 0.981 |
| All | 0.557 | 0.679 | 0.835 |

- The head path helps: 18 % less error than time only, 33 % less than copying. With 15 frames
  given the error is 0.511.
- It is far from exact in a location never seen: about half of the variance is left after 1 s.
  After a turn of more than 90 degrees the error is 0.83, close to the average frame.
- Within the first half second nothing beats copying, and copying already costs 0.27. The class
  token changes that much between neighbouring frames, which is a floor for any mismatch signal.
- Both models overfit the training locations: the training loss keeps falling (0.33 with poses at
  the end) while the held-out error is flat from step 2,500. So prediction in a known location may
  be much better than these numbers; not measured yet.
- The start-up check passed: re-anchoring at token 0 moves the cached poses by 1.5e-05 at most.

### Known against new locations (18 Nov 2026, job 13297099, `gray/evaluate.py`)

Same models, one frame given, all horizons; up to 8,192 windows per set. `runs/features-v1/evaluate.json`.

| Set | With poses | Time only | Copy | With poses / copy |
|---|---:|---:|---:|---:|
| Windows trained on | 0.360 | 0.491 | 0.815 | 0.44 |
| Training location, unseen recording (57 sequences) | 0.414 | 0.550 | 0.734 | 0.56 |
| Loc_08, unseen recording (1 sequence, a workout) | 0.279 | 0.357 | 0.425 | 0.66 |
| New location (109 sequences) | 0.558 | 0.677 | 0.829 | 0.67 |

- Knowing the location helps: 0.414 against 0.558, and an unseen recording in a known location
  is close to the training windows (0.360). Most of the gap seen in training is the location,
  not memorized windows.
- The unseen recordings are the ones left out for a weak body-clock fit, with more Workout and
  Simon says than average, so their copy error is lower too. Relative to copying, the known
  location still does better (0.56 against 0.67).
- The single Loc_08 recording is a workout in front of a television; it says little about Loc_08.

### Mismatch along recordings (same job, `gray/trace.py`)

Prediction made 1 s earlier from 15 frames, at every frame of four recordings.
`runs/features-v1/trace/` has the traces, plots and sheets of frame pairs.

| Recording | Median | 99 % | Max |
|---|---:|---:|---:|
| Having a meal, new location | 0.31 | 0.65 | 1.82 |
| Game night (table football), new location | 0.23 | 0.90 | 1.53 |
| Where is X, new location | 0.50 | 1.25 | 1.97 |
| Workout, Loc_08 | 0.20 | 0.70 | 1.25 |

What the peaks show, from looking at the 16 largest per recording:

- Overall, the largest are fast turns of 60 to 110 degrees into parts of the room not seen in the
  last second, and doors, cupboards or the fridge filling the view.
- With the head nearly still, they are the wearer changing the scene (closing the fridge, opening
  a drawer or the washing machine, pulling the shower curtain), another person walking through,
  and the wearer's own arms, hands or hair in front of the camera.
- So the signal does react to real changes, and peaks are 3 to 6 times the median. But the
  wearer's own body in view produces peaks of the same size, which a comparator would have to
  explain away. In the full model the planned body motion is known and could do that.
- "Head nearly still" only tests yaw and position. Looking up or down is not excluded, and some
  of those peaks are a change of pitch.
- Cut windows (77 tokens) and whole windows (120) agree on the same frames to within 3 %.

Not answered: whether a threshold separates the events one wants (a blocked path, a moved object)
from the rest. That needs labelled events or a rendered scene with inserted objects.

## Experiment 2: the joint model

`gray/joint.py`, job `jobs/joint.sbatch`, local check `tests/smoke_joint.py`. Built 19 Nov 2026;
not trained on Viper yet.

- **Token.** Head (9), feature (384), motion (32 pose latents and 4 root channels), all
  standardized and in the frame anchored at now (`frames.anchor`, `frames.anchor_root`).
- **Network.** One plain transformer (width 512, 8 layers) over the 120 frame tokens and the CLIP
  text tokens. Each part's noise level is embedded and added to its token.
- **It predicts the clean value, not the velocity.** With velocity as the target, the made-up
  test did not learn the features at all: the output would have to reproduce 384 values of
  noise per token. Sampling turns the prediction into a velocity.
- **Training.** Now is one of the first 30 tokens; everything up to now is clean. In 25 % of the
  clips the body motion up to now is not given (a cold start). The later frames are noised by
  stage: head (25 %), feature given the head (25 %), motion given both (25 %), feature given
  head and motion (10 %), three independent levels (15 %). Text is dropped in 10 %.
- **The fourth stage** is not part of the agreed order. It is there to measure whether knowing
  the body motion makes the features more predictable (the wearer's arms in view, experiment 1).
- **Comparator.** One step from pure noise gives the expected feature along a given head path.
  It is the same quantity experiment 1 regressed, so its error can be read against 0.557
  (one frame given) and 0.511 (15 frames given).
- **Evaluation during training**, held-out locations, warm start (1 s of history) and cold start
  (one frame, no body): the whole chain sampled (head and pelvis error against staying in
  place), the expected feature along the real head path with and without the real body, and
  the motion sampled along the real head path and features. Joint errors in cm need STEER's
  codec and are left for a separate script.
- **Not in this experiment:** the replanning loop itself (trigger, stop, continue from the last
  tokens). It needs this model first.

### Result (19 Nov 2026, job 13305584, 2 h 23 min on one MI300A)

100,000 steps, batch 128, 27.5 M parameters. 1,024 held-out windows (6 held-out locations), all
horizons. Log `/ptmp/enazaren/gray/runs/gray-joint-13305584.log`, run `runs/joint-v1/`.

| Test | Warm start (1 s given) | Cold start (1 frame, no body) |
|---|---:|---:|
| Expected feature along the real head path (MSE) | 0.477 | 0.524 |
| The same with the real body motion given | 0.477 | 0.522 |
| Copying the frame at now | 0.815 | 0.840 |
| Experiment 1, same quantity (4,096 windows) | 0.511 | 0.557 |
| Motion along real head path and features: pelvis | 7.0 cm | 8.5 cm |
| Motion along real head path and features: body yaw | 10.7° | 15.3° |
| Whole chain, one sample: head position | 50.8 cm | 63.2 cm |
| Staying in place: head position | 52.2 cm | 56.5 cm |
| Whole chain, one sample: pelvis | 43.3 cm | 56.9 cm |

- The feature part is a little better than experiment 1's dedicated model (0.477 against 0.511).
- Giving the body motion does not change the feature error. The class token's expected value
  does not become more predictable from the body.
- Given the real head path, the pelvis is placed to 7 cm. Placement was STEER's main error.
- One sampled head path is as far from the recorded one as staying in place (closer only in the
  first second: 10 against 13 cm), and its heading is further off (40° against 34°). One sample
  against one recording cannot tell a wrong path from another valid one; that needs several
  samples per window and a look at whether the paths follow the text.
- Held-out losses were still falling slowly until about step 85,000 (feature 0.294, head 0.087).

### Head paths: wrong, or valid but different? (19 Nov 2026, job 13329680, `gray/paths.py`)

376 held-out windows with narration, 10 head paths each, warm start; head position error in cm
over all horizons (4 to 8 s in brackets). `runs/joint-v1/paths.json` has the cold start too.

| | cm |
|---|---:|
| One draw, real text | 50.8 (74.9) |
| Best of 10 draws | 29.4 (39.7) |
| Expected path (one step, no sampling) | 39.8 (58.5) |
| Staying in place | 51.8 (74.2) |
| Distance between two draws | 44.5 (66.5) |
| One draw, no text | 55.6 (83.3) |
| One draw, text of another window | 69.8 (106.1) |

- The paths are valid but different: the best of 10 is far closer than staying, and the model's
  expected path beats staying by 23 %. One draw scores like staying only because the draws are
  44 cm apart from each other.
- The text is used: the wrong text is worse than none, none is worse than the real one, and
  with the same noise another text moves the path by 48 cm.
- How far the head ends up from now is predicted with a correlation of 0.82 (recorded 0.86 m,
  generated 0.78 m), so the model knows whether the person walks or stays.
- The narration describes what happens in the window, so the text tells the model about the
  future, as a prompt would. Best of 10 is lower than one draw by selection alone; read it
  against the spread.

## Experiment 3: the replanning loop along recordings

`gray/loop.py`, job `jobs/loop.sbatch`. Built 19 Nov 2026.

- **One planning step** (`loop.plan`): with the last second given, generate head path, features
  and motion for the next 5 s, and predict the expected feature of every frame. On a recording
  the prediction is made along the real head path; deployed, along the planned one.
- **Mismatch:** mean squared difference between the predicted and the observed feature,
  averaged over 0.2 s. The prediction is the one made when the plan was made, so its age grows.
- **Test:** for every now on a 0.2 s grid of a recording, the plan and the prediction are
  computed once. Triggers are then replayed on these tables at many settings: fixed interval,
  random, mismatch over a threshold, mismatch relative to what is usual at the plan's age, and
  an oracle that sees the plan's real head error. They are compared by the head and pelvis
  error of the plan in use, at the same number of replans per minute.
- **Limits.** Windows are cut to 6 s so every now lies in a cached window. The camera follows
  the person, so a replan does not change what happens next; the test only asks whether a
  mismatch marks the moments where a plan has gone stale. Stopping, and the loop in a rendered
  scene, are not part of it.

### Result (19 Nov 2026, job 13332434, 5 min)

12 recordings in the six held-out locations, 183 minutes, 10 sampling steps per part. Head
position error (cm) of the plan in use, by replans per minute. `runs/joint-v1/loop/`.

| Trigger | 15 | 20 | 30 | 45 | 60 |
|---|---:|---:|---:|---:|---:|
| Fixed interval | 27.1 | 21.5 | 15.8 | 11.5 | 9.0 |
| Random moments | 29.7 | 26.5 | 21.2 | 15.8 | 12.0 |
| Mismatch over a threshold | 26.9 | 22.1 | 17.1 | 13.1 | 10.2 |
| Mismatch relative to the plan's age | 29.2 | 25.1 | 20.2 | 14.7 | 12.0 |
| Oracle (sees the head error) | 19.6 | 14.7 | 10.5 | 7.8 | 6.5 |

- The mismatch trigger does not beat fixed intervals: equal at 15 replans per minute, 5 to 14 %
  worse above. It is clearly better than random moments, so it is not noise, but most of that is
  the plan's age, which the raw mismatch grows with: the age-corrected one is no better than random.
- The oracle is about 30 % better than fixed intervals, so well-timed replans would help. The
  mismatch does not find those moments.
- A likely reason: on a recording a plan goes stale because the person takes another of the
  valid paths (44 cm between two draws, see above), and that is not something the camera shows
  along the real head path. A deployed loop executes its own plan, so this cause does not exist
  there; what is left for the comparator is a world that differs, and recordings cannot test it.
- Pelvis errors follow the same order (fixed 11.8, mismatch 13.2, oracle 8.8 cm at 30 per minute).
- Cut and whole windows agree: expected head error and mismatch differ by under 1 % on the same frames.

## The rendered scene (looked up 19 Nov 2026, nothing built)

Both open points are answered by STEER's reconstruction work on MPI (`~/projects/STEER/nymeria_reconstruction`,
data under `/scratch/inf0/user/enazaren/nymeria/loc08_scene` and `.../splats`; not on Viper, not committed).

- **Two renderable scenes of Loc_08.** A painted mesh scene (annotated walls, floors, doors and
  ShapeR furniture meshes, coloured from the head cameras; `loc08_scene/paint_v2/scene_painted.glb`,
  rendered with pyrender) and Gaussian splats (`splats/loc08`, `splats/loc08_v2`).
- **Objects can be moved or added in the mesh scene.** Every piece of furniture is its own mesh
  with an id (83 fixed, 14 that moved between recordings). In a splat that is much harder.
- **Rendered and real features differ a lot** (`loc08_scene/dino/dino_compare.json`, 3,239
  held-out poses, cosine of the DINOv3 class token to the real frame): next real frame 1 s later
  0.80, painted scene 0.45, splat 0.44, a real frame from elsewhere 0.34. With people masked:
  0.54 and 0.55. Blurring the real frame by 8 px alone gives 0.66, so missing detail is the main
  cause; people, the missing ceiling and tone are smaller.
- **Decision (user, 19 Nov):** inside the simulation the comparator works on features of
  rendered images only. That needs the model to predict rendered features, so the scene has to
  be rendered along the recorded Loc_08 head paths, encoded with DINOv3, and the joint model
  trained or fine-tuned on those features in place of the real ones.

## The head path first (from 19 Nov 2026)

Decision (user, 19 Nov): get the first stage right before going further, the head path from
text and the egocentric view. Two standing rules came with it:

- **Robot inputs only.** The model gets nothing a robot would not have: camera image, text, its
  own pose. No gaze, and the head-path runs get no human body motion as history.
- **Indoor navigation.** Standing around and outdoor walks are not what the model is for.

What follows was built for that, in this order. Where things are: `README.md`.

### Pinhole image features (`gray/rectify.py`, `gray/pinhole.py`; 19 Nov, job 13346903)

The cache's features are the DINOv3 class token of the whole Aria fisheye frame (112 degrees
across, 1408 x 1408 in `preview_rgb.mp4`, stored upright), squeezed to 224 x 224. A robot camera
is a pinhole camera, so the features were recomputed on a pinhole view cut out of every frame with
the recording's lens calibration: along the Aria camera axis, 79.6 degrees wide and high (focal
600 at 1000 px, the camera of the rectified frames the Loc_08 reconstruction was built from),
cut out at 896 px and averaged down to 224.

- 1,074 recordings, 17.7 M frames, 13 GB in `/ptmp/enazaren/gray/pinhole`.
- Check: the fisheye frame encoded again matches the cache (cosine 0.996 median, 0.988 worst).
  Pinhole against fisheye feature of the same frame: cosine 0.80.
- The stored video is the sensor image turned clockwise in every recording (measured from gravity).

The Unitree G1 was looked at and set aside for now: its RealSense D435i colour camera is 69 x 42
degrees and, by Unitree's robot description (`d435_joint`, pitch 0.831 rad), mounted 47.6 degrees
below level on a head without a neck. That view is recorded completely by the fisheye in only 21
to 60 % of the poses of two recordings checked (`gray/views.py`). Not verified on a robot.

### Navigation subset of windows (`gray/census.py`; 19 Nov, jobs 13350161 to 13368416)

Of 499,834 windows, 49 % stay within 0.5 m of where they start and 35 % get 1 m or further.
Text and movement agree: of windows whose text does not mention walking, 4 % get 1 m or further.
About 104,000 windows name a destination (a verb of going, a direction word and a thing).

The subset keeps every window that gets 1 m or further and a random 12 % of the others, indoor
only: no outdoor activities (hike, fresh air); with a room map, at least 80 % of the recorded
head path over annotated floor; without a map, only activities labelled indoor. 161,964 windows;
with the usual split 134,974 for training and 19,121 held out. File:
`/ptmp/enazaren/gray/runs/census/navigation.npz`.

### Room maps (`gray/layout.py`; 19 Nov, job 13368415)

NymeriaPlus annotates walls, floors, doors, windows, stairs and furniture as 3D boxes for 953 of
the 1,100 recordings; recordings of one location share a world frame. STEER's
`scenebench/scene/nymeria.py` merges them (walls with door openings cut out, furniture as it stood
in each recording). `gray/layout.py` draws that on a 5 cm grid seen from above, one per storey:
floor, wall (0.3 to 1.5 m above the floor), furniture with its kind, stairs. 939 recordings,
435,217 windows, 74 MB.

What looking at wall crossings of recorded paths showed (`gray/crossings.py`, with video frames):
none was a real wall. They were doorways and closet doors without a door box, an attic without
annotated floor above the ground-floor walls, stairs, and a roof deck. Hence:

- every point is looked up on the storey its head height belongs to; near annotated stairs walls
  are not checked; a head that fits no annotated floor is "unmapped" and not checked;
- where a recorded pelvis path passes through a wall, a doorway of 0.4 m is opened (3.4 % of the
  wall cells). Recorded paths then touch no wall by construction, so the test of the maps is the
  control: recorded head paths turned by 90 or 180 degrees touch a wall in 33 and 36 % of the
  subset windows (53 and 55 % of those that get 2 m or further).

Furniture is not a hard obstacle for a head: recorded heads are inside beds, couches, chairs and
tables (sitting, leaning). All locations with all paths: `gray/atlas.py`, pictures in
`/ptmp/enazaren/gray/atlas` (45 locations, 209 km of head path).

### Scoring a head path without the recorded one (`gray/navigate.py`, `gray/score.py`)

Several valid paths exist for one text, so distance to the recorded path says little. On
held-out subset windows with a map, 8 paths are drawn per window from 1 s of history, and scored:
touches a wall; enters furniture other than beds, couches, chairs and tables; gets 1 m or
further, against whether the text mentions walking; ends within 1 m of an object the text names,
where the person really walks up to it from 1.5 m or further. Controls: the text of another
window, the recorded path, the recorded path turned by 90 degrees.

Results on the same 3,000 held-out windows (20 Nov; `joint-v1` job 13378328, `head-v1` jobs
13378326 and 13378327). `head-v1` is `gray/joint.py` with only the head path in the loss, trained
on the subset with pinhole features and no body history.

| Windows that get 1 m or further (2,331) | joint-v1 | head-v1 | recorded | recorded, turned 90 deg |
|---|---:|---:|---:|---:|
| Touches a wall | 18.4 % | 20.9 % | 0 % | 40.3 % |
| Gets 1 m or further | 66.9 % | 79.6 % | 90.3 % | |
| Distance to the recorded path, one draw / best of 8 | 107.6 / 63.6 cm | 105.9 / 62.0 cm | | standing still 125.8 cm |

| Gets 1 m or further when the text... | joint-v1 | head-v1 | recorded |
|---|---:|---:|---:|
| mentions walking | 67.8 % | 77.9 % | 83.0 % |
| does not | 4.0 % | 7.6 % | 6.7 % |

Destination (64 windows): ends within 1 m of the named object in 33.6 % (joint-v1) and 32.0 %
(head-v1) of the draws, against 20.3 % and 25.0 % with another window's text.

Reading: the subset makes the model move as often as people do; neither model avoids walls
(one path in five), which one summary vector per frame cannot be expected to teach.

### The image grid (`gray/patches.py`; 20 Nov, job 13408001)

The patch tokens of the same DINOv3 pass on the pinhole view, 14 x 14 averaged 2 x 2 to a 7 x 7
grid of 384 numbers, kept for the frames that can be "now" in a subset window (the first 30 frames
of each). The model attends to the grid of the frame at now like it attends to the text; the
class token stays the thing that is predicted for the comparator. `head-v2` is `head-v1` plus the
grid (jobs 13408002, 13408003); result below when it is in.

### Moments: samples that start where a sentence starts (`gray/tracks.py`, `gray/moments.py`, `gray/head.py`, `gray/score.py`; 20 Nov)

Weaknesses of the 8 s windows for a model that is to be prompted:

- the text of a window is every sentence overlapping it, so depending on where "now" falls a
  sentence may describe what already happened or what starts seconds later;
- history is at most 2 s and usually less; the horizon varies between 6 and 8 s;
- whether a window "goes" is measured from its first frame, not from now.

Nymeria's narration describes a recording in consecutive pieces of 5 s, one sentence each (median
34 words). A moment is one such piece: now is the frame where the sentence begins, the text is
that sentence, and the 5 s it describes are what is predicted, from the 2 s before. In training
now is moved by -5 to +15 frames and the history is cut to a random length.

- `gray/tracks.py` writes the head path of each recording as one track, read from the trajectory
  as the cache did (checked against the cache's windows), in a ground frame with +y up.
- `gray/moments.py` lists the moments with the same labels and the same indoor navigation rule as
  the windows, measured from now, and holds out the same locations.
- `gray/head.py` is a model for the head path only: past poses and image features, the sentence,
  optionally the image grid; flow model that predicts the clean path.
- `gray/score.py` is `gray/navigate.py` for moments.

### Collisions: all obstacles, not only walls (user, 20 Nov)

The aim is paths that collide with nothing. So far the trusted measure was walls, because a head
is legitimately inside furniture when its owner sits or leans. `gray/score.py` therefore counts

- **touches a wall**: any point of the path in a wall cell;
- **moves through furniture**: any point inside furniture of any kind while the head moves faster
  than 0.5 m/s. Sitting down and leaning over a counter are slow, walking through a sofa is not;
- **collides**: either of the two.

The recorded paths give the floor of each measure (walls: 0 by construction of the maps;
furniture: whatever people really do). What the maps do not have cannot be checked: other people,
pets, things that were moved during a recording, small objects on the floor. `gray/navigate.py`
(the windows) still reports the older "enters hard furniture" measure.

## Review by another model, and the plan that follows (20 Nov 2026)

The user had the repository reviewed by another model. Its points, and what was decided.

**Not adopted as stated**

- "The image grid already helps in head-v2 (90.5 against 96.1 cm in the training log)." Those two
  numbers are not comparable: for grid models the in-training check gives no body history, for
  head-v1 it still did, although head-v1 never saw body history in training. Only `navigate.py` on
  the same windows compares them.
- "The grids do not cover moments." True when written; the all-frames run (`patches.py --all`,
  job 13426460, folder `grids/`) removes the restriction.

**Done right away**

- Text guidance at sampling (`gray.head.sample(guidance=...)`, `score.py --guidance`): each step
  uses  without text + guidance x (with text - without text). No retraining; training already
  drops the text for 10 % of the samples.
- Generated rotations are made proper rotations in `gray.frames.matrix` (Gram-Schmidt).
- The in-training "goes" rule in `gray/head.py` now is the scoring rule (gets 1 m or further).
- Watch training against held-out loss on moments: 40,000 steps are about 130 passes over
  39,130 moments.

**Next, in this order**

1. A probe (`gray/probe.py`): can the distance to the nearest obstacle in 16 directions be read
   off the image features, with the room maps as truth? For the class token and for the 7 x 7
   grid. If a small probe cannot read obstacles from the features, a generator will not avoid them.
2. One model on moments with: grids of several past frames, each cell tagged with the direction
   it looks in (known camera, relative pose); the local free space or obstacle distances from the
   maps as an extra output during training only; short goal prompts ("go to the sofa") built from
   the destinations `gray/census.py` extracts, mixed with the narrations; training on all indoor
   moments before the navigation subset if the small set overfits.
3. The body as a separate model given the head path (Experiment 2: pelvis 7 cm from the recorded
   one when the head path is given), after the head path avoids obstacles.

**Open, the user's decision**

- The comparator. The review argues that predicting the class token cannot tell "the world
  changed" from "I turned into a room I had not seen", and proposes (a) to re-check the old plan
  against the new image with the head-path model and replan when the model wants to change it,
  and (b) to compare predicted and observed features only along the planned path, with a
  predicted uncertainty. (b) stays with Gray's sensory prediction; (a) does not, and may also
  fire when the model merely prefers another valid path. Both can be tested once the head path is good.
- The closed-loop test in Loc_08 with four cases (clear path, turning into an unseen room, an
  object moved into the path, an object moved off the path). Agreed in principle; rendered images
  give features that differ from real ones (cosine 0.45), which has to be solved first.

### The comparator as a tracking filter (20 Nov)

A second note from the review, adopted as the working view of option (b) above. A Kalman filter
predicts the next measurement, takes the difference to the real one, and judges that difference
relative to how uncertain the prediction was. The comparator does the first two and thresholds
the raw difference. What is missing is the uncertainty:

- turning into a room that was not in view gives a large difference that is expected (the
  feature error reached 0.83 after turns of more than 90 degrees in new rooms): no alarm;
- a change in a place that was in view gives a difference against a confident prediction: alarm.

Consequences: predict an uncertainty with the feature and threshold the difference divided by it;
compare in a space with little variation that does not matter for walking (features along the
planned path, obstacle distances) rather than the class token of the whole image; and separate
two reactions, a routine replan on a clock (Experiment 3: as good as replanning on mismatch) and a
stop when a confident prediction is contradicted.

Limits: it only helps where confident predictions are actually good (the error floor is 0.27 at
half a second); a learned uncertainty has to be checked against the real error; with 2 s of
history little counts as "seen"; behind a corner the possible views are several different ones,
which one uncertainty value blurs.

First test, nothing trained (`gray/spread.py`, `jobs/spread.sbatch`): the joint model draws the
feature of each future frame several times along the real head path; the variance between the
draws is its uncertainty. Does it rise with the real error, with turning and with time, and is
the error divided by it steadier than the raw error? Recordings have no unexpected changes, so
this shows only whether expected surprises are removed. Result below when it is in.

#### Result of the uncertainty test (20 Nov, job 13427832, `joint-v1`, 1,024 held-out windows, 8 draws)

The spread between draws follows the real error closely: rank correlation 0.86 over 107,520 frames.

| Head has turned since now | Mismatch | Spread | Mismatch / spread |
|---|---:|---:|---:|
| 0 to 15 deg | 0.333 | 0.159 | 2.36 |
| 15 to 45 deg | 0.518 | 0.270 | 2.09 |
| 45 to 90 deg | 0.669 | 0.358 | 1.98 |
| 90 deg and more | 0.749 | 0.414 | 1.87 |

- The raw mismatch is 2.25 times as large after the largest turns as after the smallest; divided
  by the spread it is 0.79 times. The expected surprise of turning is removed (slightly more than
  removed). The same holds over time since now (ratio 2.45 in the first second, 2.09 after 4 s).
- The model is too sure of itself: a calibrated spread would give a ratio near 1, it is about 2,
  and nearly 3 where the spread is smallest.
- The ratio is only a little steadier than the raw mismatch (90th percentile 1.63 against 1.92
  times the median, 99th 2.59 against 2.72), so a threshold on it would not be much sharper yet.

So an uncertainty signal is in the model without training for it, and it removes the effect of
turning. Whether the ratio reacts to a real change of the world cannot be seen on recordings.

#### Result of the probe (20 Nov, job 13427833, `gray/probe.py`; 118,313 training frames, 17,545 from held-out locations)

Mean absolute error in metres of the distance to the nearest obstacle, directions inside the
camera's view (ahead and 22.5 degrees to either side):

| Probe | Walls only (mean distance 2.80 m) | Walls or furniture (1.39 m) |
|---|---:|---:|
| Constant (median per direction) | 1.35 | 1.04 |
| Class token, linear / hidden layer | 0.81 / 0.78 | 0.79 / 0.75 |
| Grid 7 x 7, linear / hidden layer | 0.71 / 0.65 | 0.68 / 0.67 |

To the sides and behind every probe stays near the constant (walls: 1.03 to 1.14 against 1.23 and
1.21), as it must, since that is out of view.

Reading: the features do carry the room, and the grid carries more than the class token, but
not much more (0.65 against 0.78 m for walls, 0.67 against 0.75 m with furniture), and an error
of two thirds of a metre is too coarse to steer around a chair. A single 7 x 7 grid without any
geometry is therefore not expected to make the head path avoid obstacles by itself. This supports
step 2 of the plan above (several frames, each cell tagged with where it looks, obstacle distances
as a training target). Limits of the probe: it is small and briefly trained, and it is not told
how far the camera is tilted down.

## Getting the image features right: the test bed (from 20 Nov 2026, `gray/probe.py`)

Decision (user, 20 Nov): iterate on the image features until something lets the room be read
off them, before training more generators. `gray/probe.py` is the test bed: a fixed set of
frames with their pinhole images (30 per recording, 939 recordings, 27,976 frames, 16 GB in
`/ptmp/enazaren/gray/probeset`), the true distance to the nearest obstacle in 16 directions from
the room maps, and small probes fitted per candidate feature. A candidate costs minutes.

### Round 1 (jobs 13430206 to 13433316)

Two flaws of the test were found by looking at frames (`probe.py look`) and fixed first:

- A head inside the footprint of a sofa or over a table made every distance zero. The piece of
  furniture the camera is in or over no longer counts for that frame.
- Depth Anything V2 (metric, indoor) gives distances that are too large for our wide view; the
  best single factor on the training locations is 0.55.

Error in metres inside the camera's view on held-out locations (24,438 training frames, 3,538
held-out), and how well "an obstacle within 1 m straight ahead" is told apart (AUC):

| Features | Walls | Walls or furniture | Obstacle ahead | ...where the truth is under 1 m | 1 to 2.5 m | 2.5 m and more |
|---|---:|---:|---:|---:|---:|---:|
| Constant | 1.35 | 1.12 | 0.50 | 0.71 | 0.57 | 2.71 |
| Camera tilt and height only | 1.26 | 1.07 | 0.66 | 0.75 | 0.52 | 2.51 |
| Class token | 0.81 | 0.98 | 0.77 | 0.79 | 0.73 | 1.71 |
| DINO grid 7 x 7 | 0.77 | 0.91 | 0.80 | 0.79 | 0.80 | 1.30 |
| Depth (large) | 0.64 | 0.85 | 0.83 | 0.60 | 0.56 | 1.78 |

DINO at 14 x 14 and at double resolution are within noise of 7 x 7; the three depth sizes of each
other. Reading obstacle distances off the depth by geometry, without a probe, gives 0.98 m for
walls or furniture after scaling (1.8 m before).

Reading:

- Every feature tells open space from closed space: where the next obstacle is 2.5 m or further
  away the error halves against the constant. That is all they tell. Where an obstacle is closer
  than 2.5 m, which is three quarters of the in-view directions, no probe beats the constant.
- 44 % of the in-view directions have an obstacle within 1 m. People stand at counters and
  tables; such an obstacle is below the camera's view unless the person looks down. A single
  frame cannot show it, whatever the feature.
- The probes are also short of data: the 7 x 7 grid scored 0.65 (walls) with 118,000 frames and
  0.77 with 24,000.

So the choice of feature is not the bottleneck in this test; the single frame and the target are.
Next: score only obstacles that can be in view (needs the height of each piece of furniture in
the maps), and give the probe several frames with their poses.

### Round 2: only what the camera can see, and earlier frames (20 Nov, jobs 13447338 to 13447398)

The maps now store the lowest and highest point of every piece of furniture. A third truth,
"seen", counts an obstacle only if some point between its bottom and top falls inside the image.

- The explanation given after round 1 was wrong. Of the in-view directions with an obstacle
  within 1 m, the camera can see the obstacle in 95 %. Scored on "seen" the probes are no better
  than on all obstacles (constant 1.14, class token 0.99, DINO grid 0.85, depth 0.85 m).
- Earlier frames do not help the probe. On 94,669 training and 14,040 held-out frames that have a
  7 x 7 grid now, 1 s ago and 2 s ago:

| Probe | Walls | Walls or furniture | ...where under 1 m | Obstacle ahead (AUC) |
|---|---:|---:|---:|---:|
| Constant | 1.32 | 1.09 | 0.68 | 0.50 |
| Grid of now | 0.64 | 0.73 | 0.61 | 0.87 |
| Grids of now, 1 s and 2 s ago, with camera poses | 0.66 | 0.74 | 0.61 | 0.86 |

With four times the frames the one-frame grid probe is clearly better than in round 1 (0.73
against 0.91 m), so the probes of round 1 were limited by data. What stays: for obstacles closer
than 1 m no probe, not even one on metric depth, is better than the constant by more than 0.1 m.

Open question: is that the features, or is the truth itself only accurate to about half a metre
at close range? The furniture is annotated as boxes, which enclose empty space (the legs of a
table, the gap behind a chair), and things move. Next check: the recordings come with the
semidense point cloud of Aria's SLAM, real measured points in the same world frame. Obstacle
distances from those points, against the distances from the boxes, show how far the maps can be
trusted at close range, and could replace them as the truth.

### head-v2: windows with the 7 x 7 grid of the frame at now (20 Nov, jobs 13408002, 13447325)

Same 3,000 held-out windows as joint-v1 and head-v1 above.

| Windows that get 1 m or further (2,331) | head-v1 | head-v2 | recorded |
|---|---:|---:|---:|
| Touches a wall | 20.9 % | 18.2 % | 0 % |
| Gets 1 m or further | 79.6 % | 81.2 % | 90.3 % |
| Distance to the recorded path, one draw / best of 8 | 105.9 / 62.0 cm | 101.8 / 58.9 cm | |

Goes when the text mentions walking 79.3 % (head-v1 77.9 %), when it does not 7.7 % (7.6 %).
Destination (64 windows): ends within 1 m of the named object in 41.0 % of the draws (head-v1
32.0 %), with another window's text 23.2 % (25.0 %).

Reading: the grid helps a little everywhere and most where an object is named, but one path in
five still touches a wall. Training took 4 h 43 min against 2 h 23 min without the grid.

### head-v3: the head path on moments (20 Nov, jobs 13447315 to 13447321 on the test partition, `gray/head.py`, `gray/score.py`)

39,130 training moments, pinhole class token per past frame, no image grid, 40,000 steps (about
50 min of training, run as 15-minute pieces that resume). 3,000 held-out moments with a room map, 8 draws.

| Moments that go 1 m or further (2,158) | Generated | With another moment's text | Recorded | Recorded, turned 90 deg |
|---|---:|---:|---:|---:|
| Touches a wall | 17.6 % | 17.8 % | 0 % | 39.5 % |
| Moves through furniture | 37.1 % | 31.9 % | 20.2 % | 57.4 % |
| Collides (either) | 45.6 % | 40.2 % | 20.2 % | 69.8 % |
| Gets 1 m or further | 88.3 % | 69.0 % | 100 % | |

Distance to the recorded path: one draw 75.5 cm, best of 8 43.8 cm, standing still 112.9 cm.

| Gets 1 m or further when the sentence... | Generated | Recorded |
|---|---:|---:|
| mentions walking (2,208) | 85.9 % | 92.0 % |
| does not (792) | 13.6 % | 15.9 % |

Destination (58 moments): ends within 1 m of the named object in 53.7 % of the draws, at least
one of 8 draws in 84.5 %, with another moment's text 24.4 %; median distance at the end 0.87 m
(start 1.97 m, other text 1.76 m).

Reading:

- Aligning text and now works. The model goes when the sentence says so nearly as often as
  people do, and it reaches a named object in half of the draws against a third to two fifths on
  windows (head-v1 32 %, head-v2 41 %; different samples, so only roughly comparable).
- It does not avoid obstacles better: 17.6 % of the paths touch a wall.
- The furniture measure has a high floor: recorded paths "move through furniture" in 20 % of the
  moments that go. Either the 0.5 m/s rule is too loose or the boxes are larger than the
  furniture; the point-cloud check proposed above would tell.
- Mild overfitting: the held-out loss is flat from step 25,000 (0.116 to 0.117) while the
  training loss keeps falling (0.104 to 0.087); the sampled paths still improve a little until 35,000.

Text guidance at sampling (`score.py --guidance`, jobs 13450934, 13450935) does not help:

| Guidance | Goes when told | Collides (moments that go) | Destination | ...at least one of 8 |
|---|---:|---:|---:|---:|
| 1 (none) | 85.9 % | 45.6 % | 53.7 % | 84.5 % |
| 2 | 89.4 % | 49.5 % | 52.6 % | 75.9 % |
| 3.5 | 91.3 % | 52.9 % | 51.7 % | 69.0 % |

It makes the model go more often, but into more obstacles, no closer to the named object, and
with less variety between draws. Left at 1.

### head-v4: moments with the 7 x 7 grid of the frame at now (21 Nov, jobs 13426460 and 13426479)

The grids now exist for every frame of every recording (`/ptmp/enazaren/gray/grids`, 621 GB,
1,074 recordings). head-v4 is head-v3 plus the grid of the frame at now; same 3,000 held-out moments.

| Moments that go 1 m or further (2,158) | head-v3 | head-v4 | Recorded | Recorded, turned 90 deg |
|---|---:|---:|---:|---:|
| Touches a wall | 17.6 % | 14.5 % | 0 % | 39.5 % |
| Moves through furniture | 37.1 % | 33.9 % | 20.2 % | 57.4 % |
| Collides (either) | 45.6 % | 41.0 % | 20.2 % | 69.8 % |
| Gets 1 m or further | 88.3 % | 87.5 % | 100 % | |
| Distance to the recorded path, one draw / best of 8 | 75.5 / 43.8 cm | 71.5 / 41.5 cm | | standing still 112.9 cm |

Goes when the sentence mentions walking 85.5 % (head-v3 85.9 %), when it does not 12.0 %
(13.6 %). Destination (58 moments): 50.4 % of the draws end within 1 m (53.7 %), with another
moment's text 22.4 % (24.4 %). Held-out loss 0.1147 (0.1174), still flat from step 30,000.

Reading: as on windows (head-v1 to head-v2: 20.9 to 18.2 %), the grid takes about three points
off the wall rate and a little off everything else, and leaves text following and destinations
where they were. It is worth keeping and it is not the fix.

### The room maps against Aria's measured points (21 Nov, `gray/cloud.py`, job 13467373, 24 held-out recordings)

Aria's SLAM stores measured 3D points per recording (171 M in these 24, 44 % within the
recommended uncertainty limits). Counted on the maps' grid at body height, and compared where
the head came within 3 m.

A first run showed that the cloud contains the wearer: 84 % of the recorded head positions lay in
cells "occupied by points" (arms, hands, body seen by the cameras). Cells within 0.3 m of the
recorded head path are therefore cleared. Points on other people and on the wearer's arms beyond
0.3 m remain; there is no observation time per point on disk to filter them by.

| Check | Result |
|---|---|
| Outline cells of furniture boxes with a measured point within 10 / 20 / 30 cm | 76 % / 84 % / 88 % |
| Wall cells with a measured point within 10 / 20 / 30 cm | 72 % / 84 % / 90 % |
| Cells occupied by points that are further than 20 cm from any wall or furniture of the map | 33 % |
| Recorded head positions moving faster than 0.5 m/s that are inside a furniture box | 10.7 % |
| In-view directions of test-bed frames where map and points agree on the nearest obstacle within 0.3 m | 29 % (median difference 0.60 m) |

Reading:

- The boxes are where the furniture is: three quarters of their outline have a measured point
  within 10 cm. The maps are not misplaced.
- The maps are incomplete or the cloud is not clean: a third of what the points show at body
  height is not near anything in the map. Part of that is clutter and unannotated things, part
  is people. The two cannot be separated with the data on disk.
- The boxes include space that people walk through: one moving head position in nine is inside
  a furniture box.
- As a yardstick for the distance to the nearest obstacle the two sources agree poorly, but the
  cleared corridor and the remaining points on people both distort exactly that number, so it
  does not settle which is right.

Conclusion: walls are a trustworthy measure (the turned-path control, and probes read them to
0.64 m at a mean of 3 m). Furniture boxes are good enough to say roughly where furniture is and
not good enough to score or train close-range avoidance to a few decimetres. The point cloud as
stored cannot replace them, because of the people in it. Not run on all 118 held-out recordings;
the 24 answer the question as far as this data can.

### How accurate is the estimated depth? (21 Nov, `gray/cloud.py depth`, jobs 13477713 and 13478085)

Question (user): with a depth map, could furniture hits be judged more accurately than with the
boxes? Only if the depth is accurate, and the Aria glasses have no depth sensor, so it would be
estimated. Test: Depth Anything V2 (metric, indoor) against the points Aria measured, projected
into 3,538 test-bed frames of the 118 held-out recordings (2.5 M cells of the 28 x 28 depth with
at least five points; the near end of the points in a cell counts as the visible surface).

| Depth model (large) | Median error | 0.5 to 1 m | 1 to 2 m | 2 to 4 m | Within 10 % | Within 25 % |
|---|---:|---:|---:|---:|---:|---:|
| One scale factor for all frames | 0.26 m | 0.17 | 0.28 | 0.52 | 26 % | 53 % |
| One factor per recording | 0.26 m | 0.16 | 0.28 | 0.54 | 26 % | 54 % |
| One factor per frame | 0.23 m | 0.13 | 0.24 | 0.50 | 32 % | 59 % |

- The model's depth is 2.4 times too large for our view (measured / estimated 0.42), and the
  factor varies from frame to frame (0.32 to 0.56 for 10 to 90 % of the frames).
- Even with the best factor per frame, which needs the measured points, half of the cells are
  off by more than a fifth. The small and base models are within 0.03 m of the large one.
- Reading the depth as distance along the line of sight instead of along the optical axis is worse (0.30 m).
- The measurement is rough itself: points on people are in the cloud, a cell is 2.8 degrees wide,
  and points behind the visible surface can be the nearest ones measured. The true error is
  somewhat smaller than these numbers, not several times smaller.

Conclusion: this depth is no finer a yardstick than the boxes, so no depth-based collision
measure is built. As an input it was the best single-frame candidate in the test bed, but by a
small margin, and this explains why.

## A truth without the furniture boxes, and the feature comparison redone (21 Nov 2026, `gray/ground.py`)

Decision (user, 21 Nov): a known map of the scene is not an acceptable input. And since the
furniture boxes are too rough, the feature comparison against them says little about furniture
and has to be redone on a better truth.

`gray/ground.py` builds that truth per location from all its recordings together:

- **free**: within 0.2 m of a pelvis position that moved faster than 0.4 m/s while the head was
  more than 1.2 m above the floor. People do not walk through things. Two looser versions were
  tried and rejected by looking at the picture (`ground.py --show`): every place a head was, and
  head positions moving fast; both mark sofas, beds and counters as free, because heads are over
  them when sitting and sweep over them when leaning.
- **blocked**: the walls of the room maps, and cells at body height in which Aria's SLAM measured
  points in at least 30 % of the location's recordings (at least 3). Furniture that stands still
  repeats; people do not, except on the paths everybody takes, which are free.
- everything else is unknown.

45 locations, `/ptmp/enazaren/gray/ground/<location>.npz`; the well-measured points of each
recording are kept in `ground/points/`. In the picture of Loc_08 the free cells are the corridors
between the furniture and the repeatedly measured cells fill the furniture boxes.

For a frame, "walkable" is how far the free cells reach along a direction and "blocked" where the
first blocked cell begins; the true free distance lies between. Where they agree within 0.3 m
their mean is the target "free"; that holds for 94 % of the in-view directions of the held-out
test-bed frames (mean 1.43 m, under 1 m in 52 %).

### Features on the free distance (jobs 13481583 to 13481587; 24,438 training and 3,538 held-out frames)

Error in metres inside the camera's view, and telling "free for less than 1 m straight ahead" (AUC):

| Features | All | Under 1 m | 1 to 2.5 m | 2.5 m and more | Blocked ahead (AUC) |
|---|---:|---:|---:|---:|---:|
| Constant | 1.16 | 0.47 | 0.99 | 2.96 | 0.50 |
| Camera tilt and height only | 1.08 | 0.58 | 0.72 | 2.62 | 0.68 |
| Class token | 0.88 | 0.74 | 0.67 | 1.47 | 0.84 |
| DINO grid 7 x 7 | 0.82 | 0.69 | 0.68 | 1.27 | 0.84 |
| Depth (large) | 0.74 | 0.42 | 0.52 | 1.71 | 0.87 |
| Depth (large) with camera tilt | 0.71 | 0.44 | 0.46 | 1.62 | 0.89 |

Reading:

- On a truth that can be trusted the candidates separate. Depth is clearly best where it
  matters, within 2.5 m: 0.44 and 0.46 m against 0.69 and 0.68 m for the DINO grid.
- The DINO features tell open from closed (best beyond 2.5 m) but place a near obstacle worse
  than the constant does (0.69 to 0.74 against 0.47 m under 1 m).
- The grid is hardly better than the class token within 2.5 m.
- Depth improves with the camera tilt, as geometry should; the DINO features do not.
- All probes are short of data (24,438 frames), so the absolute numbers understate every candidate.

So the feature that is best for avoiding furniture is depth, not a finer DINO grid.

### Round 2 on the free distance (21 Nov, jobs 13488552 to 13488555)

| Features (24,438 training frames) | All | Under 1 m | 1 to 2.5 m | Blocked ahead (AUC) |
|---|---:|---:|---:|---:|
| Depth Anything V2 large, with camera tilt | 0.71 | 0.44 | 0.46 | 0.89 |
| ZoeDepth, with camera tilt | 0.75 | 0.61 | 0.49 | 0.87 |
| Depth and DINO grid together, with camera tilt | 0.81 | 0.66 | 0.66 | 0.83 |

| DINO grid 7 x 7 (94,669 training frames) | All | Under 1 m | 1 to 2.5 m | Blocked ahead (AUC) |
|---|---:|---:|---:|---:|
| Constant | 1.09 | 0.61 | 0.68 | 0.50 |
| Grid of now | 0.63 | 0.47 | 0.50 | 0.89 |
| Grids of now, 1 s and 2 s ago, with camera poses | 0.63 | 0.49 | 0.50 | 0.89 |

- A second depth model is no better than the first.
- Giving the probe depth and grid together is worse than depth alone: with this little data the
  larger input overfits. A statement about the probe, not about the features.
- The amount of data matters as much as the feature: the DINO grid with four times the frames
  (0.63 m, AUC 0.89) is level with depth on the small set. At equal data depth is ahead; depth on
  the large set has not been measured, because depth exists only for the test-bed frames.
- Earlier frames still add nothing in a probe.

Where this stands: the best probes tell "blocked within 1 m ahead" at 0.89 and place the end of
free space to about half a metre. That is coarse, not good enough to say the features separate
free space from furniture. The probes are small and short of data, so the generator itself,
trained on all moments with the free distance as an extra target, is the next and better test.

### Scoring paths on the new truth (`gray/score.py`, jobs 13489903 and 13489905)

The furniture-box measure is replaced. A path "moves through something blocked" if a point lies
in a blocked cell of `gray/ground.py` while the head moves faster than 0.5 m/s, and the share of
its moving points that lie where people are known to have walked is reported.

| Moments that go 1 m or further (2,158) | Recorded | Recorded, turned 90 deg | head-v3 | head-v4 |
|---|---:|---:|---:|---:|
| Touches a wall | 0 % | 39.5 % | 17.6 % | 14.5 % |
| Moves through something blocked | 7.3 % | 71.3 % | 40.8 % | 36.1 % |
| Moving points where people walked | 98.3 % | 55.3 % | 81.3 % | 84.1 % |

The measure separates far better than the boxes did (recorded 7 % against turned 71 %; with the
boxes 20 % against 57 %), and it shows the size of the problem: more than a third of the
generated paths go through something, where the wall rate alone suggested one in seven. The
7 % on recorded paths are heads over blocked cells while moving (leaning, reaching).

### head-v5: the free distance as a training target (21 Nov, job 13504033)

head-v4 plus one extra token that has to say, during training only, how far the way is free
around the camera at now (16 directions, walked distance and distance to the first blocked cell,
from `gray/ground.py`). The token reads the past frames, the text and the image grid but not the
future path; the path tokens can read it. Same 3,000 held-out moments.

| Moments that go 1 m or further (2,158) | head-v4 | head-v5 | Recorded | Recorded, turned 90 deg |
|---|---:|---:|---:|---:|
| Touches a wall | 14.5 % | 13.4 % | 0 % | 39.5 % |
| Moves through something blocked | 36.1 % | 33.7 % | 7.3 % | 71.3 % |
| Moving points where people walked | 84.1 % | 85.3 % | 98.3 % | 55.3 % |
| Distance to the recorded path, one draw / best of 8 | 71.5 / 41.5 cm | 69.5 / 42.5 cm | | |

Destination (58 moments): 58.0 % of the draws end within 1 m of the named object (head-v4
50.4 %), at least one of 8 in 89.7 %, with another moment's text 24.8 %; median distance at the
end 0.76 m. Goes when the sentence mentions walking 85.5 %, when not 12.3 %.

The model's own free distance, inside the camera's view, on held-out locations: off by 0.57 m
(walked) and 0.61 m (blocked) from step 5,000 to the end. On the training moments it falls to
0.28 m (all directions).

Reading:

- A small gain on every obstacle measure and a larger one on destinations, so the target is kept.
- The free distance does not generalise: 0.28 m on locations the model was trained on, 0.57 m on
  new ones, and no better than the small probes (0.63 m for the same grid). There are only 39
  training locations; the model learns their rooms, not how to read a room from an image. More
  steps cannot fix that. A feature that already carries geometry from many scenes can, which is
  the case for the depth below.

### Depth for all frames (`gray/depth.py`, job 13503360)

Depth Anything V2 (metric, indoor, base) on the pinhole view of every frame, 28 x 28 cells:
1,074 recordings, 26 GB in `/ptmp/enazaren/gray/depth`. Against the depth stored in the test bed
(other code, frame by frame): median difference 0.9 %, at most 1.5 % in any recording.
head-v6 is head-v5 plus this depth for the frame at now: the depth of each cell of the image
grid, and one token with the distance to the nearest thing at body height in 16 directions, read
off the depth with the camera's tilt and height (`gray.probe.from_depth`), both scaled by 0.42.

### head-v6: estimated depth as input (22 Nov, job 13518809)

head-v5 plus the depth of the frame at now (per cell of the image grid, and as 16 distances at
body height read off the depth). Same held-out moments.

| Moments that go 1 m or further | head-v5 | head-v6 |
|---|---:|---:|
| Touches a wall | 13.4 % | 13.0 % |
| Moves through something blocked | 33.7 % | 34.1 % |
| Moving points where people walked | 85.3 % | 85.5 % |
| Free distance in view, held-out (walked / blocked) | 0.57 / 0.61 m | 0.55 / 0.60 m |
| Destination: ends within 1 m | 58.0 % | 57.3 % |

No difference. The estimated depth adds nothing to what the model already gets from the DINO
grid, on the paths or on the free distance. With the probes (depth ahead at equal, small data;
the grid level with it at four times the data) this says the two carry much the same
information about the room once there is enough data, and that neither gets the free distance on
new locations below about 0.55 m. head-v5 stays the reference; the depth files are kept.

### Where the collisions are, and choosing among draws (22 Nov, `gray/score.py`, job 13533129, head-v5)

| Moments that go 1 m or further (2,158): moves through something blocked | Recorded | Recorded, turned 90 deg | Generated |
|---|---:|---:|---:|
| within the first second | 1.1 % | 21.9 % | 2.1 % |
| within the first two seconds | 2.9 % | 41.4 % | 7.9 % |
| within the 5 s | 7.3 % | 71.3 % | 33.7 % |

| How the draw is chosen among 8 | Moves through something blocked | Gets 1 m or further |
|---|---:|---:|
| Any draw | 33.7 % | 87.6 % |
| The draw that fits the free distances the model itself predicts | 19.7 % | 81.6 % |
| The best draw (needs the truth: a ceiling, not a method) | 3.7 % | |

Reading:

- The collisions are late in the path. In the first second a generated path goes through
  something hardly more often than a recorded one (2.1 against 1.1 %); the excess builds up
  from the second second on. A system that plans again every second executes mostly the part
  that is already good. This is what the closed loop is for.
- Among 8 draws there is nearly always one that goes through nothing (3.7 %, below the 7.3 % of
  the recorded paths). The model can produce a clear path; it cannot tell which one it is.
- Its own free-distance estimate, rough as it is (0.57 m), already picks better: 19.7 against
  33.7 %, at the cost of going a little less often. Choosing among draws is the cheapest lever
  found so far, and a better chooser has room up to the ceiling.

## The comparator on moments (from 22 Nov 2026, `gray/compare.py`)

Decision (user, 22 Nov): the head path is good enough to move on. `head-v5` is the planner: over
the first second, which a system that plans again executes, its paths go through something
hardly more often than recorded ones (2.1 against 1.1 %); reaching the quality of the best of 8
draws over 5 s would need the room reading that 39 training locations do not give. The two
explorative runs (`head-x4`, `head-x8-one`, jobs 13565237 and 13565238; best of K noises per
sample, after Gladstone, Ji, Du, arXiv 2607.27372) and a chooser among draws stay side tasks.

The next component is the comparator, and it stays with Gray's idea: predict what the camera
will show, compare with what it shows. Decided with it:

- **What is predicted** stays the class token of the pinhole view, one vector of 384 numbers per
  frame. It is what exists for every frame and what the earlier experiments used. Its known
  weakness (it changes with arms, light and a television as well as with the room) is to be
  measured, not assumed; a coarse grid is the fallback.
- **With its uncertainty.** The model predicts, per value, a variance (Gaussian likelihood). The
  signal is the "surprise": the squared error divided by the predicted variance, averaged over the
  384 values, about 1 when the world is as expected. Turning into an unseen room gives a large
  error and a large variance, so no alarm; the same error where the model was sure is one.
- **Along the real path.** The poses of the frames to come are an input: at run time the head's
  real poses, so that a mismatch means the world differs and not that the plan was left.
- One pass, no sampling: the comparator has to run at every frame.

`compare-v1` trains this on the navigation subset of moments (the same inputs as `head-v5`: past
poses and class tokens, the sentence, the image grid of the frame at now).

### Explorative training, and the number of sampling steps (22 Nov, jobs 13565237, 13565238, 13570710, 13571815, 13571816)

Tried on the user's suggestion: Explorative Modeling (Gladstone, Ji, Du, arXiv 2607.27372). Per
training sample K noises are tried and only the one whose prediction is closest to the recorded
path is trained on (`head.py --explore K`); `--one-step` always starts from pure noise, so a path
is generated in one pass. `head-x4`: head-v5 with K = 4. `head-x8-one`: K = 8, one pass.
Moments that go 1 m or further, same 3,000 held-out moments:

| Model, sampling steps | Touches a wall | Moves through something blocked | ...in the first second | Destination | Best of 8 draws blocked |
|---|---:|---:|---:|---:|---:|
| head-v5, 20 steps | 13.4 % | 33.7 % | 2.1 % | 58.0 % | 3.7 % |
| head-v5, 4 steps | 11.3 % | 28.5 % | 1.3 % | 61.6 % | 4.8 % |
| head-v5, 1 step | 8.8 % | 26.0 % | 0.9 % | 62.5 % | 13.9 % |
| head-x4, 20 steps | 17.7 % | 40.5 % | 2.8 % | 47.8 % | 3.8 % |
| head-x4, 4 steps | 14.6 % | 33.6 % | 1.6 % | 52.2 % | 2.2 % |
| head-x8-one, 1 step | 11.8 % | 32.9 % | 1.1 % | 57.3 % | 12.8 % |

(Recorded paths: 0 %, 7.3 %, 1.1 %.)

- Explorative training does not help here. Each explorative model is worse than head-v5 sampled
  with the same number of steps.
- What does help was found by the comparison it called for: fewer sampling steps. head-v5
  sampled in one step goes through something in 26.0 % of the moments against 33.7 % with 20
  steps, touches a wall in 8.8 against 13.4 %, and is at the level of the recordings in the first
  second (0.9 against 1.1 %). It still goes (84.5 % against 87.6 %) and reaches the named object
  as often (62.5 %).
- The price is variety. One step gives nearly the model's average path for the moment: among 8
  draws the best is hardly better than any (13.9 %). With 20 steps the draws differ widely and
  one of them is nearly always clear (3.7 %), but a single draw is worse. Four steps are in
  between (28.5 %, ceiling 4.8 %, 19.6 % when the draw is chosen by the model's own free distances).

Consequence: when one path is drawn and executed for a second, one to four steps are better
and cheaper. Twenty steps are only worth it together with a chooser among draws.

### compare-v1 and compare-v2 (22 Nov, jobs 13568021 and 13570825)

`compare-v1` (navigation subset, 39,130 moments) memorised: its held-out loss was lowest at step
5,000 and had doubled by step 30,000, the surprise rose from 1.15 to 1.99 and the rank
correlation of uncertainty and error fell from 0.80 to 0.67.

`compare-v2` trains on all moments of the training locations with now anywhere in the sentence
(`--everything --wide 37`), with dropout 0.1, and keeps the state with the lowest held-out loss
(`best.pt`, step 12,500 of 40,000). On 1,024 held-out moments at that state:

| | compare-v2 | Copying the last seen frame |
|---|---:|---:|
| Mismatch | 0.529 | 1.069 |
| Surprise (1 is calibrated) | 1.10 | |
| Rank correlation of uncertainty and error | 0.84 | |

| Head has turned since now | Mismatch | Surprise |
|---|---:|---:|
| 0 to 15 deg | 0.34 | 1.09 |
| 15 to 45 deg | 0.52 | 1.08 |
| 45 to 90 deg | 0.68 | 1.11 |
| 90 deg and more | 0.75 | 1.12 |

The mismatch more than doubles with turning and the surprise stays at 1.1: the expected surprise
of turning is removed, which is what the uncertainty is for. Later states drift (surprise 1.24,
1.15 to 1.37 across turns at step 40,000), so `best.pt` is the one to use.

What this does not show is that the surprise notices a changed world; recordings have none.

### Does the comparator notice a changed world? (23 Nov, `gray/alarm.py`, job 13592317, `compare-v2`)

4,000 held-out moments in 5 locations (90 recordings). The prediction for the frame 1 s after
now is compared with the true frame and with three others; "above the level" is the share that
exceeds what 5 % of the true frames exceed.

| Observed frame (1 s after now) | Moments | Mismatch: median | told from true (AUC) | above the level | Surprise: median | told from true (AUC) | above the level |
|---|---:|---:|---:|---:|---:|---:|---:|
| True | 4,000 | 0.31 | | 5 % | 1.03 | | 5 % |
| Revisit: same pose, same recording, 10 s or more apart | 2,956 | 0.41 | 0.71 | 11 % | 1.49 | 0.78 | 39 % |
| Other day: same pose, another recording of the home | 3,036 | 0.63 | 0.84 | 35 % | 1.88 | 0.86 | 57 % |
| Elsewhere in the recording | 4,000 | 1.07 | 0.87 | 68 % | 2.80 | 0.95 | 80 % |

At 2 and 4 s the order is the same and everything is closer together (elsewhere above the level
in 68 and 61 % by the surprise, 59 and 54 % by the mismatch).

- The order is right for both signals: true, revisit, other day, elsewhere.
- The surprise is the sharper signal for a different scene (80 against 68 % at 1 s) and for
  another day (57 against 35 %).
- It is also sharper where it should not be: a revisit of the same place, where nothing changed,
  exceeds the level in 39 % of the cases (mismatch 11 %). Either "nearly the same pose" (0.25 m,
  15 degrees) is too loose to give the same view, or the class token changes with things that do
  not matter for the path (the wearer's hands, people, what is being handled). Not separated yet.
- Another day is only moderately above a revisit (1.88 against 1.49), so a moved piece of
  furniture is not yet something this signal can be trusted to pick out from ordinary variation.

#### Tighter pose match, and a floor under the variance (23 Nov, jobs 13593771, 13593772, 13593970)

Is the revisit alarm rate a fault of the loose pose match? No. With "nearly the same pose" at
0.09 m and 5 degrees, and at 0.04 m and 2 degrees, the raw mismatch of a revisit falls to the
level of the true frame (alarm rate 4 % and 1 %), but the surprise still alarms in 28 % and 26 %.
A revisit this close is mostly a person standing still. There the model is very sure, its
predicted variance is small, and the small differences between two views of the same place
(hands, people, what is being handled) are large relative to it.

A tracking filter divides by the predicted variance plus the noise of the measurement itself. That
second term was missing. It is measured here as half the mean squared difference between a
revisit and the true frame, per value, on one half of the moments (0.189 on average; the
predicted variance is 0.393 on average) and applied to the other half. 1 s after now, share
above the level that 5 % of the true frames exceed:

| Observed frame | Moments | Mismatch | Surprise | Surprise with floor |
|---|---:|---:|---:|---:|
| Revisit (nothing changed) | 1,964 | 4 % | 28 % | 16 % |
| Other day | 727 | 24 % | 57 % | 50 % |
| Elsewhere | 4,000 | 69 % | 79 % | 79 % |

The floor nearly halves the false alarms on revisits and keeps the detection. It does not bring
them to 5 %. The raw mismatch is the quietest on revisits and the weakest at detecting; the
surprise with floor is the best compromise of the three so far.

## Second review, and the free checks it suggested (23 Nov 2026)

A second review by another model proposed, among others: less starting noise when sampling; a
scorer that rates a drawn path by the image cells along it; dropout, masking of grid cells and
training on all moments for the head path; left-right mirroring; training the comparator with
revisits as targets; judging the alarm over several frames; the strongest cells on the path
instead of their mean; and comparing geometry instead of appearance. Three checks cost nothing
and were run first (user, 23 Nov). The rest is listed under "next" in `STATUS.md`.

### Less starting noise when sampling (`sample(temperature=...)`, jobs 13596626 to 13597383)

head-v5, moments that go 1 m or further, same 3,000 held-out moments:

| Steps, noise factor | Touches a wall | Goes through something | ...in the first 2 s | Best of 8 | Destination | Gets 1 m or further |
|---|---:|---:|---:|---:|---:|---:|
| 20, 1.0 | 13.4 % | 33.7 % | 7.9 % | 3.7 % | 58.0 % | 87.6 % |
| 4, 1.0 | 11.3 % | 28.5 % | 5.7 % | 4.8 % | 61.6 % | 86.7 % |
| 20, 0.7 | 10.6 % | 24.3 % | 4.6 % | 3.5 % | 59.3 % | 85.5 % |
| 4, 0.7 | 9.7 % | 21.2 % | 3.8 % | 4.9 % | 61.9 % | 84.9 % |
| 4, 0.5 | 8.9 % | 17.5 % | 3.1 % | 4.9 % | 61.0 % | 84.1 % |
| 4, 0.3 | 8.5 % | 15.2 % | 2.6 % | 6.6 % | 61.2 % | 83.9 % |
| 4, 0 (no noise: one path per moment) | 8.0 % | 13.9 % | 2.2 % | 13.9 % | 62.1 % | 83.5 % |

Recorded: 0 %, 7.3 %, 2.9 %.

- The less noise, the fewer collisions, down to 13.9 % with none, against 33.7 % as sampled
  before. The path the model produces without noise is its best single guess; noise spreads the
  draws, and mostly into worse ones.
- Text following and destinations are unchanged within noise.
- Variety goes with the noise: at 0.5 the best of 8 draws is still 4.9 %, at 0.3 it is 6.6 %, and
  without noise all draws are the same.
- Over the first two seconds a factor of 0.5 or less is at the level of the recordings.

Planner setting from here on: head-v5, 4 steps, noise factor 0.5 (17.5 %; variety kept for a
later chooser). Without a chooser, 0.3 or 0 is better still.

### Judging the alarm over five frames (`gray/alarm.py`, job 13597381, compare-v2)

Averaging the surprise with floor over five consecutive frames (a third of a second) does not
help: a revisit alarms in 16 % against 14 % for the single frame, another day in 48 against
46 %, elsewhere in 80 against 78 %. What sets a revisit apart from the true frame lasts longer
than a third of a second.

### Which description of the image to compare (`gray/spaces.py`, job 13596630)

No predictor: the distance between a frame and its revisit, the same pose on another day, and a
frame from elsewhere, in several descriptions (7,080 frames of the held-out homes, 3,619 with a
revisit, 1,110 of those with another day; pose within 0.09 m and 5 degrees).

| Description | Revisit | Other day | Elsewhere | Other day told from revisit (AUC) | Elsewhere told from revisit (AUC) |
|---|---:|---:|---:|---:|---:|
| Class token | 0.358 | 0.632 | 1.035 | 0.75 | 0.83 |
| Grid 7 x 7 | 0.365 | 0.568 | 1.227 | 0.72 | 0.91 |
| Grid, lower middle of the image | 0.393 | 0.664 | 1.222 | 0.72 | 0.88 |
| Depth | 0.106 | 0.144 | 0.333 | 0.64 | 0.88 |
| Depth, scale removed | 0.098 | 0.134 | 0.274 | 0.64 | 0.86 |
| Depth, scale removed, lower middle | 0.090 | 0.123 | 0.243 | 0.61 | 0.81 |

- A different scene is told from a revisit best by the grid (0.91).
- Depth is disturbed least by another day (0.64). Whether that is good cannot be read from this:
  "another day" mixes changes that matter for walking with changes that do not, and a description
  that ignores it is right about the second kind and wrong about the first. A test with known
  moved furniture (Loc_08) is needed to decide.

### compare-v3: the image grid as what is predicted (23 Nov, jobs 13594455 and 13602196)

`compare.py --target grid`: the 7 x 7 grid of every fifth frame, each cell reduced to 64 principal
components (73 % of the variance of the cells), with a variance per value; otherwise as
compare-v2. 20,000 steps, 3 h 5 min, no overfitting (held-out loss still falling at the end).
On held-out moments: mismatch 0.526 against 1.156 for copying the grid of the frame at now
(compare-v2: 0.529 against 1.069), surprise 1.08, rank correlation of uncertainty and error
0.86; across turns of 0 to more than 90 degrees the mismatch rises from 0.40 to 0.69 and the
surprise from 1.08 to 1.11. One output token per frame is enough to carry the grid.

Alarm test (3,000 held-out moments, pose within 0.09 m and 5 degrees; 26 % of the cells are on
the path on average). Share above the level that 5 % of the true frames exceed, 1 s after now:

| Observed frame | compare-v2, surprise with floor | compare-v3: mismatch | surprise with floor | on the path only | 3 strongest cells on the path |
|---|---:|---:|---:|---:|---:|
| Revisit (nothing changed) | 16 % | 3 % | 16 % | 13 % | 13 % |
| Other day | 50 % | 25 % | 45 % | 41 % | 37 % |
| Elsewhere | 79 % | 80 % | 91 % | 85 % | 80 % |

At 2 s and 4 s the revisit rate of the surprise with floor is 9 % and 6 %, with elsewhere at
85 % (2 s).

- The grid is the better thing to predict: a different scene is caught in 91 % against 79 %.
- Restricting the comparison to the path helps little: revisits 13 against 16 %, and detection
  falls with it. The three strongest cells are no better. So the false alarms on revisits do not
  come from the edges of the image.
- They are a matter of the first second: where the model is most sure, two views of an unchanged
  place differ by more than it allows for. A single floor for all situations does not capture
  that. Training with revisits as targets, so that the model's own variance learns it, is the
  direct remedy and the next thing to try.
- The raw mismatch of the grid alarms on a revisit in 3 % and on a different scene in 80 %, but
  it rises with turning, which is what the variance was introduced against.

## Three things at once (from 23 Nov 2026, afternoon)

The user chose to pursue together: finishing the comparator, the body motion, and more homes.

### Pictures of generated head paths (`gray/show.py`, job 13602888)

Twelve held-out moments, six draws each with 4 steps and half the starting noise, on the ground
maps (`runs/head-v5/show/`, copies on the workstation in `gray/paths/`). Paths follow corridors
and the gaps between furniture. Typical failures: stopping short of where the person went, and
the wrong direction where the sentence says little about where to go. The six draws stay close
together, as expected with reduced noise.

### Revisits as targets for the comparator (`gray/revisits.py`, job 13603987; `compare-v4`, job 13603988)

For every frame a frame of the same recording at least 10 s away from nearly the same pose
(9 cm, 5 degrees of heading, 3.5 of tilt, 10 cm of height): 8.5 M of 17.7 M frames have one (48 %).
`compare.py --revisits` replaces half of the target frames by their revisit, so that the
predicted variance has to cover how an unchanged place varies. The first lookup job hung for its
15 minutes without output: forked workers stalled in their first torch call on Viper. Workers
are now spawned. Results: pending.

### More homes: Ropedia Xperience 2 (`gray/ropedia.py`, jobs 13604983 to 13605340)

Question: is reading a room limited by the 39 training homes? Dataset: arXiv 2611.17060,
Hugging Face `ropedia-ai/Ropedia-Xperience-2`, CC BY-NC 4.0. The paper announces 1,500+ scans and
4,000+ recordings; online on 23 Nov was a first batch of 126 scenes. Its task recordings have no
images, and the registration of recordings to the LiDAR scans is not published. Usable are the
head-worn walk-throughs: undistorted keyframes of four cameras, poses, and a mesh per session.
105 indoor ones were downloaded (49 GB, `/ptmp/enazaren/ropedia`). Only 41 have a surface mesh;
the other 64 are an older export (camera markers and sparse points in `mesh.ply`, and a
calibration file that states only the sensor).

From the 41: 14,946 views of 79.6 degrees cut from the two front cameras, and the free distance
from the session's mesh by the rules of `gray/ground.py` as far as one walk-through allows (see
the module's text). Checked by eye on six views (`ropeset/look.png`).

The probe of `gray/probe.py` on the 7 x 7 grid, mean error of the free distance in view (m):

| Trained on | Held-out Nymeria homes: all | under 1 m | 1 to 2.5 m | 2.5 m and more | Held-out Ropedia scenes: all |
|---|---:|---:|---:|---:|---:|
| The median distance | 1.16 | 0.47 | 0.99 | 2.96 | 0.74 |
| Nymeria, 24,438 views of 39 homes | 0.82 | 0.69 | 0.68 | 1.27 | 0.65 |
| Ropedia, 13,346 views of 27 scenes | 1.00 | 0.60 | 0.74 | 2.20 | 0.51 |
| Both | 0.81 | 0.64 | 0.68 | 1.33 | 0.52 |

Giving the probe the camera's tilt and height changes nothing (0.82 to 0.83). With equal numbers
of views the picture is the same.

- Adding the Ropedia scenes does not improve reading held-out Nymeria homes: 0.81 against 0.82.
- The two datasets transfer poorly to each other: a probe trained on one is clearly worse on the
  other than a probe trained there (1.00 against 0.82; 0.65 against 0.51). For far distances a
  Ropedia-trained probe is hardly better than the median on Nymeria (2.20 against 2.96, Nymeria-trained 1.27).
- One sign in the other direction: for obstacles nearer than 1 m the Ropedia-trained probe is
  better on Nymeria than the Nymeria-trained one (0.60 against 0.69), and both together give 0.64.
- So the test does not answer the question. 27 scenes from another camera rig, with another
  definition of the truth, are too few and too different to show whether hundreds of homes
  would help. The held-out Ropedia part is 4 walk-throughs, so its column is rough.
- What would make it a fair test: the rest of the dataset when it is published (with the
  registration to the LiDAR scans, all walk-throughs could be used), or a dataset of posed indoor
  images with meshes that is large today (ARKitScenes, ScanNet++), at the price of a hand-held camera.

### The body as one track per recording (`gray/body.py`, jobs 13604865 and 13604987)

The cache holds the body per 8 s window, each with its own floor (the lowest joint of the
window), so overlapping windows agree only where their floors do. The body was encoded again per
recording from the motion capture, in the cache's representation (272 numbers per frame at 30
per second, MoLingo's autoencoder, 32 numbers per image frame), with the track's floor smoothed
over 20 s, and with the root path (pelvis, root yaw) kept in the track's ground frame. All
1,074 recordings; 77 to 90 % of the frames of a recording have a body. Against the cache on the
same frames the latents differ by 0.01 to 0.11 (median per recording) at a size of about 2.5;
single windows by up to 2.7, taken to be the cache's floor steps (not inspected). The head joint
is 14 to 15 cm from the camera everywhere.

### The body model (`gray/motion.py`, `motion-v1`, job 13605449)

Given the past 2 s (head, image, body), the head path of the next 5 s, the sentence and the image
grid at now, it generates the body of those 5 s (pose latent and root path, 36 numbers per frame),
as a flow model like `head.py`. The past body is withheld in 30 % of the samples (cold start).
Scores on held-out homes: pelvis distance against "straight under the camera", root yaw, latent
error, warm and cold. `tests/smoke_motion.py` passes. Results: pending.

### motion-v1: result, and the checks before the loop (23 Nov, jobs 13606041 to 13606054 on the test partition, `gray/chain.py`, job 13611133)

Trained 40,000 steps in fourteen 15-minute jobs that resume one another. On 1,015 held-out moments,
one generated body per moment along the recorded head path (20 steps, full noise):

| | With the past body | Without it |
|---|---:|---:|
| Pelvis from the recorded one | 5.6 cm | 7.1 cm |
| Root yaw | 8.9 degrees | 12.3 degrees |
| Latent error (1.0 is the average pose) | 0.56 | 0.69 |

The pelvis placed straight under the camera is 22.2 cm off. The held-out loss was flat at the
end (0.0664, 0.0665).

Checks on 755 held-out moments that have a body for 11 s. Decoding was verified on the recorded
body first: its head joint is 13.5 cm from the camera, as measured when the tracks were built.

**Sampling.** With the past body:

| Steps | Starting noise | Pelvis | Yaw | Latent | Head joint to camera |
|---:|---:|---:|---:|---:|---:|
| 20 | 1.0 | 5.7 cm | 8.8 | 0.56 | 13.3 cm |
| 4 | 1.0 | 5.1 cm | 8.0 | 0.51 | 13.1 cm |
| 4 | 0.3 | 4.7 cm | 7.3 | 0.47 | 13.0 cm |
| 4 | 0.0 | 4.7 cm | 7.2 | 0.46 | 13.0 cm |
| 1 | 0.0 | 6.1 cm | 7.8 | 0.47 | 13.8 cm |

As for the head path, fewer steps and less noise are better and five times cheaper. One step is
worse for the pelvis. Without the past body the same order holds (6.0 cm at 4 steps and noise 0.3,
against 7.1 cm). Setting from now on: 4 steps, noise 0.3.

**The chain.** Head path from `head-v5` (4 steps, half the noise), body along it. Per second 1 to 5:

| | Head joint to camera, cm | Body facing against camera, degrees | Pelvis from under the camera, cm |
|---|---|---|---|
| Recorded body, recorded path | 13.5 13.5 13.4 13.3 13.5 | 23.6 23.3 22.2 22.8 22.1 | 22.4 22.0 22.3 22.5 22.8 |
| Generated body, recorded path | 13.2 13.2 13.0 12.8 13.1 | 22.8 22.0 20.7 21.0 20.7 | 22.3 22.3 22.7 23.0 23.4 |
| Generated body, generated path | 13.0 12.8 12.6 12.5 12.7 | 20.7 17.9 16.7 15.9 15.4 | 21.9 21.6 21.7 22.1 22.6 |

The body belongs to a generated path as it does to a recorded one: the head joint stays where
the camera is, without drift over the 5 s. Along generated paths the body faces nearer to where
the camera looks (15 to 21 degrees against 22 to 23). That is a property of the generated head
paths, which look around less than people do, not a lag of the body.

**Rolled forward** along the recorded head path, five times 1 s, continuing from its own output:

| Second | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Pelvis, rolled forward | 3.6 | 4.7 | 5.2 | 5.4 | 5.6 cm |
| Pelvis, the 5 s at once | 3.7 | 4.5 | 4.8 | 5.0 | 5.5 cm |
| Head joint to camera, rolled forward | 13.2 | 13.0 | 12.7 | 12.6 | 12.8 cm |

No drift: continuing from its own output costs nothing against generating the 5 s at once.

Not tested: rolling the head path forward as well (needs the images along a path nobody walked),
and how the motion looks; the numbers do not show foot sliding or stiff poses.

### The chain prompted in Loc_08 (23 Nov, `gray/scene_worker.py`, `gray/animate.py`, job 13611998)

A worker that runs the chain at a requested spot in one home, in the protocol of the earlier
project's scene worker, so that the Loc_08 viewer page works with it (`gray/scene_server.py`
serves the page from the workstation). Per request: the recorded frame nearest to the spot gives
the image and the 2 s before (head, images, body); `head-v5` plans the head path from the prompt;
`motion-v1` generates the body along it; the body is decoded and placed at the spot. A request
takes 0.14 s once the models are loaded. The comparator is not part of it yet.

Six prompts at random recorded spots (`--demo`), rendered as videos (`docs/media/`) and one sheet:

| Prompt | Pelvis travels | Head joint to planned camera |
|---|---:|---:|
| C walks forward. | 0.63 m | 16.4 cm |
| C walks to the kitchen. | 1.58 m | 8.8 cm |
| C turns around and walks to the sofa. | 2.21 m | 12.8 cm |
| C stands still and looks around. | 0.35 m | 9.8 cm |
| C walks to the table. | 1.82 m | 12.7 cm |
| C walks down the hallway to the bedroom. | 3.36 m | 9.5 cm |

Looked at as stills from the videos and the sheet: the figures are upright and walk along the
planned path through free space; the one told to stand stays; "walks forward" from a person
standing at a counter stays for three seconds and then moves into a blocked area. Not judged:
whether the named place is reached (the kitchen, the sofa), and foot sliding, which stills do not
show. Loc_08 is a training home, so this says how the parts work together, not how they generalise.

### On the workstation, and a better video (24 Nov, `gray/scene_server.py --local`, `gray/film.py`)

The chain runs on the workstation's RTX 3090: models, text encoder, decoder code and the data of
Loc_08's 26 recordings (15 GB) plus their videos (23 GB) were copied; the worker loads in 8 s and
answers in 0.08 s. Several prompts at one spot end in clearly different places, so the text
steers; "stands still" at a spot where the person was walking still moves 1.7 m.

`gray/film.py` makes a video per prompt: the pinhole views the models were given (`gray/seen.py`),
the home's mesh scene rendered in Blender with the planned head path and the body as an SMPL
mesh, and the scene from the planned camera poses. Where the real view shows a television or a
dresser, the render from the planned camera shows it too, so plan, scene and recording line up.

### A gallery of head paths to judge by eye (24 Nov, `gray/gallery.py`)

Per held-out sample one picture: the camera view at now and what the camera showed 1, 2.5 and
5 s later; six plans on the map, green if they go through nothing and red if they do; five other
prompts from the same view (one path each, no noise); and six paths from full noise next to the
one without. Made on the workstation from a copy of 35 held-out recordings (59 GB, 2,904 of the
samples the scores use), since the queue on Viper did not start the job for an hour.

First round, twelve random samples from four held-out homes (six recordings were local then):
plans clear in 6, 5, 5, 1, 5, 0, 6, 6, 6, 0, 3 and 6 of 6. Seen in the pictures:

- The other prompts give clearly different paths from one view: left, right, back.
- Plans leave rooms through doors and follow hallways in homes the model never saw.
- Of a long sentence with two turns, the first is followed.
- Several failures are a wall corner clipped by centimetres.
- The two samples with 0 and 1 of 6 at a staircase are not reliable: the map is flat and marks
  stairs as blocked.
- One sample goes the wrong way: the view at now shows only a table, and the plans head into
  blocked space where the person turned to the hallway.

## Motion quality first: HumanML3D (from 24 Nov 2026, afternoon)

The user set the comparator aside for now: first good motion, by also training on HumanML3D
(images replaced by a "no image" token, text as the condition, as EgoPriMo does), and a
comparison with EgoPriMo and UniEgoMotion. `BACKLOG.md` holds the open tasks.

### HumanML3D in the form of our samples (`gray/hml.py`, on the workstation)

The 272-number representation at 30 frames per second (Hugging Face `royrhea/272-dim-HumanML3D`,
8.4 GB, Apache-2.0, academic use under the AMASS licence): 26,682 motions of 2 to 10 s, half of
them mirrored copies, 54.9 h. Encoded with the same autoencoder as Nymeria's bodies (it was
trained on this data): latents of size 2.6, in the range of Nymeria's (2.2 to 2.8). The camera made
for it is 1.54 m above the floor (median) and looks 19 degrees below level, at the low end of
what Nymeria's camera does.

HumanML3D has no camera. A camera pose per frame is made from the head joint and one fixed
transform: the average of how Nymeria's camera sits on the head joint, measured on 43,920 frames
of the 61 recordings on the workstation (decoded bodies against camera tracks). The camera is
12.3 cm from the joint (10.9 cm ahead, 5.4 cm up); single frames differ from the mean by 4.4 cm
and by 17.9 degrees of rotation (median; 33 degrees at the 90 % mark), because glasses do not
follow the head joint exactly. So the head paths made for HumanML3D are smoother and more
regular than real ones.

What candidate subsets keep (a 7 s sample every second; with one given frame 5.1 s suffice):

| Rule | Motions | Hours | Samples |
|---|---:|---:|---:|
| All with a caption | 26,680 | 54.9 | 84,086 |
| 5.1 s or longer | 19,846 | 47.7 | 77,252 |
| ... and the pelvis gets 1 m or further | 9,314 | 21.6 | 33,824 |
| ... and the caption mentions walking | 8,604 | 19.9 | 30,896 |
| ... and no dancing, jumping, fighting, crawling, throwing | 7,712 | 17.7 | 27,354 |

For comparison, Nymeria's navigation subset has 39,130 training samples. Of the motions that
travel, 27 % mention turning, 11 % running, 7 % jumping, 6 % stairs.

### The body model trained with HumanML3D mixed in (`motion-v2`, `motion-v3`; 24 to 25 Nov, jobs 13656547 to 13656556)

`gray/hml.py: HmlMoments` gives HumanML3D motions as samples (27,192 samples of 7,498 motions of
its training split that are 5.1 s or longer and travel 1 m or further). The body model takes a
learned token in place of their images, and optionally in place of their head path. 30 % of the
training samples came from HumanML3D; everything else as `motion-v1`. Each run was four 1-hour
jobs on the main partition; the first ones started within a minute of submission.

Two measures of the feet that need no recorded motion were added to `gray/chain.py`: how fast a
toe moves over the ground while it is within 5 cm of its lowest height in the clip, and how high
the lower toe is above the clip's lowest point. The first is loose (recorded bodies score
25.5 cm/s, since a toe pushing off counts as down) and is read against the recorded value.

On 755 held-out moments, along the recorded head path, 4 steps and noise 0.3:

| Run | HumanML3D head path | Pelvis | without past body | Yaw | Latent | Toes slide | Lower toe above lowest |
|---|---|---:|---:|---:|---:|---:|---:|
| Recorded bodies | | | | | | 25.5 cm/s | 2.7 cm |
| `motion-v1` | no HumanML3D | 4.7 cm | 6.0 cm | 7.3 | 0.47 | 30.2 cm/s | 2.2 cm |
| `motion-v2` | masked | 5.2 cm | 6.6 cm | 7.4 | 0.47 | 30.6 cm/s | 2.1 cm |
| `motion-v3` | given | 4.7 cm | 6.1 cm | 7.0 | 0.45 | 30.6 cm/s | 2.3 cm |

- Mixing in HumanML3D did not improve the motion by any of these measures. The feet slide as
  much as before, about a fifth more than recorded bodies.
- With the head path masked, the pelvis is placed half a centimetre worse: 30 % of the samples
  no longer teach following a head path. With it given, nothing is lost and yaw and latent are
  marginally better (within what a different seed could give; not tested).
- Along generated head paths and when rolled forward, all three behave alike: the head joint
  stays 13 cm from the planned camera, no drift.
- Not shown by these numbers: how the motion looks. Pictures of the same moments from the three
  models are the next check, before concluding that HumanML3D does not help.

### UniEgoMotion as released, on its own validation set (25 Nov, on the workstation)

Its code, released weights (`uem_v4b_dinov2`), processed data (EE4D-Motion, 36 GB, with DINOv2
features per frame) and the SMPL-X model were copied from the MPI cluster to
`gray/local/uniegomotion` and `gray/local/data/ee4d`. Its environment was built with Python 3.10
(without `blendify`, which is only for rendering). Its metric scripts pointed at the authors' disk
and were pointed at the local data. Its evaluation ran on 4,405 validation clips of 8 s at 10
frames per second, about 1.5 h of sampling per task (1,000 diffusion steps) on the RTX 3090.

| Task | Given | MPJPE | after alignment | Head position | Sliding (recorded 4.36) |
|---|---|---:|---:|---:|---:|
| Reconstruction | images and head trajectory of the whole clip | 0.109 m | 0.073 m | 0.058 m | 3.58 |
| Forecasting | the first 2 s | 0.271 m | 0.094 m | 0.228 m | 2.61 |
| Generation | the first image | 0.415 m | 0.102 m | 0.382 m | 2.87 |

By 2 s segments, generation: 0.249, 0.422, 0.518 and 0.560 m. Its measures also include
penetration of the floor and time in the air. Its semantic measures need a text-motion model
at a path on the authors' disk and were not run. These numbers have not been compared with the
paper's table yet.

### UniEgoMotion's data in our form, and our two models trained on it (25 Nov, `gray/ee4d.py`, jobs 13676077 and 13676082)

`gray/ee4d.py convert` brings EE4D-Motion into the form of our samples: the SMPL-X body resampled
from 10 to 30 frames per second and encoded with the motion autoencoder, the camera path with its
axes named back to ours, and the released DINOv2 class token per frame (1024 numbers). Training
split 14,852 segments, 82.5 h; validation 4,606 segments, 25.4 h; three quarters of the segments
are 5.1 s or longer. Checks: camera heading against body facing cosine 0.93; decoding again puts
the head joint 12.2 cm from the camera (11.5 cm in their joints). No text, no image grid, no maps.

Trained on Viper with `jobs/submit.sh` (below), 40,000 steps each, done within the first 1-hour
job (32 and 34 minutes; peak memory 8.7 and 13.4 GB). On 1,024 validation samples:

| Model | Result |
|---|---|
| `ee-head-v1`, head path from the past 2 s of images and poses | one draw 26.9 cm from the recorded path (standing still: 22.2 cm); on samples that go 1 m or further 81.0 cm (standing still: 90.5 cm) |
| `ee-motion-v1`, body along the recorded head path | pelvis 4.6 cm with the past body, 6.2 cm without (under the camera: 22.7 cm); yaw 7.7 and 10.8 degrees; latent 0.67 and 0.77 |

The body model does as on Nymeria (4.7 and 6.0 cm). The head path, without text, is barely better
than standing still by distance to the recorded path, as expected where nothing says where the
person goes; this is the generation setting UniEgoMotion is in too. The comparison with
UniEgoMotion on its validation clips, first 5 s, is next.

### Running on any cluster (`jobs/submit.sh`, `jobs/clusters/`)

One settings file per cluster (repository, data folder, interpreter, GPU options, partitions,
GPU-specific settings) and one command on the workstation that asks every connected cluster when
it would start a training and submits where that is soonest; `@data` in the arguments stands for
the cluster's data folder. Viper is set up; Raven and MPI have settings files but no repository
or data yet. Trainings on Nymeria stay on Viper, where its 700 GB of features are.

### What went wrong on the workstation (25 Nov)

Two trainings started there at once, detached, with 10 loader processes each, exhausted the
memory and crashed the machine, twice. The same trainings on Viper peak at 8.7 and 13.4 GB (after
the dataset was changed to read its arrays from disk), so the cause was not the size of the data
as such; it was not found. The temporary working copy was lost with the first crash; the code now
lives in the clone on the workstation and is pushed from there. Rule since: agree with the user
where a computation runs; no training on the workstation by day.

### Our chain against UniEgoMotion on its validation clips (25 Nov, `gray/versus_uem.py`, jobs 13679602, 13680004, 13680005)

Generation from the first frame of a clip, on the 3,678 of UniEgoMotion's 4,405 validation
clips that are in our converted data and have 5 s. Both sides as the same 22 joints (theirs and
the recorded body through SMPL-X, ours decoded and moved into the clip's frame through the camera
pose of the first frame), every 0.2 s over the first 5 s. Check of the conversion: UniEgoMotion
over its 8 s gives 39.0 cm here, 37.6 cm in its own evaluation on all clips.

| | Joints | Aligned | Head | Pelvis | Pelvis travels | 1 m or more |
|---|---:|---:|---:|---:|---:|---:|
| Recorded | | | | | 0.76 m | 24 % |
| Standing still (the first frame's body, held) | 22.6 cm | 5.4 cm | 22.9 cm | 20.2 cm | 0 | 0 % |
| UniEgoMotion, released model (1,000 steps) | 33.9 cm | 8.1 cm | 33.1 cm | 31.2 cm | 0.59 m | 19 % |
| Our chain, 4 steps, noise 0.5 and 0.3 (our usual setting) | 25.4 cm | 7.0 cm | 23.6 cm | 23.1 cm | 0.22 m | 4 % |
| Our chain, 4 steps, noise 1 and 0.3 | 26.8 cm | 7.1 cm | 25.2 cm | 24.4 cm | 0.40 m | 9 % |
| Our chain, 20 steps, noise 1 and 1 (a full sample) | 32.3 cm | 8.2 cm | 30.6 cm | 29.3 cm | 0.72 m | 22 % |
| Our body along the recorded head path (usual setting) | 8.7 cm | 5.9 cm | 4.3 cm | 6.9 cm | | |
| Recorded body through our encoder and decoder | 2.9 cm | 2.6 cm | 3.9 cm | 1.8 cm | | |

- Distance to the recorded joints does not measure quality in this task: standing still beats
  every model. It rewards caution.
- With our usual low-noise sampling the chain hardly moves here (0.22 m against 0.76 m recorded):
  without text, the most likely head path is to stay, and reduced noise pulls every draw to the
  most likely path. Its lower error than UniEgoMotion's is that effect and nothing else.
- Sampled in full (20 steps, noise 1), the chain travels as people do (0.72 m, 22 % of clips
  1 m or more; toes slide 22.5 cm/s against 21.0 recorded) and is then level with UniEgoMotion
  or slightly ahead: joints 32.3 against 33.9 cm, pose after alignment 8.2 against 8.1 cm.
  UniEgoMotion travels a little less than people (0.59 m).
- So on UniEgoMotion's own data and task, our two-stage chain is about as good as its single
  model, with 20 sampling steps against 1,000. One run, one seed; the difference of 5 % in joint
  distance is not established.
- The body model is not the limit: along the recorded head path it is within 8.7 cm.
- Differences that remain: our model is given the camera's height and tilt in the first frame,
  UniEgoMotion masks its head trajectory entirely; our skeleton has average proportions (costs
  2.9 cm); 727 clips are missing on our side.
- Consequence for all generation comparisons, EgoPriMo included: report how the bodies move
  (travel, share that goes, sliding) next to any distance, with a standing-still row; and choose
  the sampling per use: low noise for a planner that should be safe, a full sample when the
  spread of what people do is what is judged.

### Our chain against the EgoPriMo reimplementation on held-out moments (25 Nov, `gray/versus_egoprimo.py`, job 13680414)

1,500 held-out moments of the navigation subset with a body and a map; both sides are given the
frame at now and the one sentence; the first 5 s; bodies decoded to the same 22 joints; the
reference is the recorded body through the same encoder and decoder. EgoPriMo: the
reimplementation of the STEER repository trained on all of Nymeria (`all-medium-v1`, its state
with the lowest validation loss, step 200,000), 50 sampling steps. Ours: `head-v5` and `motion-v1`.

| | Joints | Aligned | Head | Pelvis | Travels | 1 m or more | Through something | Toes slide |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Recorded | | | | | 1.97 m | 75 % | 3.9 % | 26.5 cm/s |
| Standing still | 83.9 cm | 8.1 cm | 86.1 cm | 81.2 cm | 0 | 0 % | | |
| EgoPriMo reimplementation | 76.5 cm | 10.2 cm | 76.4 cm | 73.3 cm | 1.49 m | 57 % | 27.1 % | 23.5 |
| Ours, only the frame at now, usual sampling (4 steps, noise 0.5 and 0.3) | 52.3 cm | 8.0 cm | 51.6 cm | 49.6 cm | 1.50 m | 63 % | 12.6 % | 28.5 |
| Ours, only the frame at now, a full sample (20 steps, noise 1) | 61.9 cm | 9.3 cm | 61.4 cm | 58.8 cm | 2.11 m | 82 % | 26.1 % | 38.2 |
| Ours, with the 2 s before now, usual sampling | 46.4 cm | 6.6 cm | 47.2 cm | 44.3 cm | 1.53 m | 64 % | 12.5 % | 29.3 |

- At the same amount of movement (1.50 against 1.49 m) our chain is a third nearer to the
  recorded body (52.3 against 76.5 cm) and goes through something half as often (12.6 against
  27.1 %). Standing still is the worst row here, since on these moments people walk: distance is
  a meaningful measure on this set.
- With text, our low-noise setting still moves (1.50 m; without text on UniEgoMotion's data
  0.22 m). A full sample overshoots on Nymeria: it travels further than people, slides more and
  goes through things as often as EgoPriMo. The usual setting stays the one to use with text.
- Limits: it is our reimplementation, not the authors' model (none was released); it is used
  outside how it was trained (8 s windows cut anywhere, all overlapping sentences, generation as
  one of three tasks); it is given the fisheye frame's feature, ours the pinhole view and the
  image grid; one run, one seed.

### How far the Nymeria planner's paths go at each sampling setting (25 Nov, `gray/travel.py`, job 13680463)

`head-v5`, 3,000 held-out moments that go 1 m or further, with the 2 s before now:

| Sampling | Path length | of recorded | Ends this far from the start | Ends 1 m or further away | Goes through something (earlier runs) |
|---|---:|---:|---:|---:|---:|
| Recorded | 2.76 m | 100 % | 2.02 m | 91 % | 7.3 % |
| 20 steps, noise 1 | 2.53 m | 92 % | 1.88 m | 82 % | 33.7 % |
| 4 steps, noise 1 | 2.38 m | 86 % | 1.84 m | 82 % | 28.5 % |
| 4 steps, noise 0.5 | 2.05 m | 74 % | 1.72 m | 81 % | 17.5 % |
| 4 steps, noise 0 | 1.96 m | 71 % | 1.69 m | 80 % | 13.9 % |

Less noise shortens the path by a fifth (2.53 to 2.05 m) but moves its end by only 9 % (1.88 to
1.72 m), and as many paths as before end 1 m or further away. So the halving of "goes through
something" is mostly paths that wander less on the way to about the same place, and only to a
small part paths that go less far. At every setting generated paths are shorter than recorded
ones; with the planner's setting by a quarter.

### The UniEgoMotion comparison with three seeds (jobs 13680004, 13680464, 13680465)

Our chain as a full sample: joints 32.3, 32.6 and 32.6 cm, travel 0.72, 0.72 and 0.73 m. Against
UniEgoMotion's 33.9 cm (one saved sample of its own). Our spread over seeds is 0.3 cm, so the
1.3 to 1.6 cm in our favour is not an accident of our sampling; whether it exceeds the spread of
UniEgoMotion's sampling is not known, since its predictions take 1.5 h per run to make.

### HumanML3D motions placed into recorded frames (25 Nov, `gray/place.py`, job 13684566)

Idea (user, 24 Nov): a HumanML3D motion gets a real image by putting its camera at now onto the
recorded camera of a training moment and keeping the pair if the motion fits the home. Checked
over the 5 s after now, every 0.2 s; 38,276 training moments with a map, 16 draws each.

| The motion goes | Draws | Not blocked | And on walked floor | And the body is clear | Height within 10 cm | All |
|---|---:|---:|---:|---:|---:|---:|
| Any distance | 612,416 | 25.2 % | 21.8 % | 18.2 % | 35.5 % | 6.6 % |
| 0 to 1 m | 19,983 | 69.2 % | 65.5 % | 58.2 % | 35.9 % | 22.2 % |
| 1 to 2 m | 118,903 | 41.8 % | 37.4 % | 32.3 % | 34.7 % | 11.7 % |
| 2 to 3 m | 220,276 | 25.7 % | 22.2 % | 18.3 % | 35.7 % | 6.7 % |
| 3 to 5 m | 218,270 | 14.5 % | 11.8 % | 9.1 % | 36.0 % | 3.2 % |
| 5 m or more | 34,984 | 6.7 % | 4.6 % | 3.1 % | 34.5 % | 1.1 % |

36,766 pairs kept (at most 4 per moment): 17,408 moments have one, 15,314 of the 27,192
HumanML3D samples are used. Kept motions go 2.15 m on average, all draws 2.90 m.

- The body rule (user, 25 Nov: every body part, not only head and pelvis): 22 joints and the 21
  bone middles against the walls of the camera's storey, and against a 3D map of what stands in
  the home (`ground/solid/`: 10 cm cubes with measured points in 30 % of the home's recordings).
- The recorded bodies of 3,036 of the frames under the same rule, as its error. First version
  (every part against the cubes, walls of any storey): 46 % passed. Of the failures, 39 % of
  bodies had a forearm or hand in a solid cube within 5 s (hands rest on things), 3 to 5 % a
  leg or the trunk; 20 % were in a wall, most of that walls of another storey. Final rule: walls
  of the camera's storey for every part (3.5 % of recorded bodies fail), cubes for all parts but
  the arms. 87.8 % of recorded bodies pass.
- The rules on the path are stricter than reality too: the recorded 5 s pass "not blocked" in
  80 % and "walked" in 76 %.
- Body points have no thickness.
- Trainings with the pairs for 30 % of the samples, otherwise as their predecessors:
  `head-v7` (as head-v5; first submitted under the name of an existing run, head-v6, where the jobs stopped at once without changing it) and `motion-v4` (as motion-v1), Viper, 1-hour jobs from
  13684906. A pair is shown as a start from a single frame.

### Scene videos against the EgoPriMo reimplementation (25 Nov, `gray/versus_scene.py`)

Eight moments drawn at random in Loc_08 (a training home of both models; the only home with a
mesh scene): recorded, ours (only the frame at now, usual sampling) and EgoPriMo in the same room,
`gray/motion/scene/` on the workstation, two of them in `docs/media/`. Joint distance to the
recorded body in the eight: ours 29 to 110 cm, EgoPriMo 58 to 209 cm, ours nearer in seven;
EgoPriMo goes through something in three, ours in none.

Note on the comparison of 25 Nov: at full noise our chain goes through something as often as
EgoPriMo (26.1 against 27.1 %) and is 61.9 against 76.5 cm from the recorded body. So most of the
difference in collisions and 10 of the 24 cm are the sampling setting, which EgoPriMo was not
given. To do: sample EgoPriMo with few steps and reduced noise too.
