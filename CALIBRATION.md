# Distance reference review — October 11, 2026

Source: `20261011222411553.MOV` in Downloads, 1920×1080, 692 frames,
27.9 seconds. The user estimates about 12 feet near the door at the start and
about 3 feet near the end, describing the latter as a personal comfort boundary.
These are approximate references, not measured distances or a validated flight
gap. The reported 60° camera FOV has an unspecified axis and was not used to
calculate distance.

## What the recording establishes

The current controller marks **every confirmed frame** as close-stop, including
the distant starting position. Its height cutoff is 65% of the image, while the
starting person box is already about 73% tall. This reproduces the user's
reported premature stop at their estimated 12-foot position.

| Reference interval | User estimate | Confirmed / total frames | Median width | Median height | Complete-body frames |
| --- | --- | --- | --- | --- | --- |
| 0–5 seconds, near door | About 12 ft | 122 / 125 | 19.0% | 73.1% | 0 |
| 20–25 seconds, near endpoint | About 3 ft | 16 / 124 | 84.1% | 98.2% | 0 |

Intervals were chosen from visual review of the recording; the user did not
provide exact timestamps. Near-end measurements describe only the 16 confirmed
frames and are not representative of uninterrupted reliable tracking.

The feet are cropped at the beginning, and the head and lower body are cropped
near the endpoint. Image height therefore cannot be treated as complete person
height for distance calculations. Width distinguishes these two views, but pose,
arms and crop can change it too. No metric distance curve was fitted and no
intermediate distances were invented.

Across all 692 frames, there were 557 confirmed targets and 135 unconfirmed
frames. The real person-mode controller returned neutral in all frames: close
stop for the 557 confirmed targets and lost-target neutral for the other 135.
These are offline frame counts, not labeled tracking accuracy or flight results.

## Calibration status

Approximate image-size references are recorded. **Flight distance calibration is
not complete.** Increasing the height cutoff alone would also leave following
blocked by its complete-body requirement, and would not solve distance sensing
from cropped bodies. Changes to those rules require a separate preview review.
The 3-foot reference is not a command to approach a face or a verified stopping
distance; live following remains disabled pending the existing control fault.

For a measured calibration, use the same camera/zoom, known marked distances and
a framing setup that shows the whole body at the intended operating distances.
If partial-body tracking is required, it needs a separately validated size
measurement rather than treating the cropped box as a full person.

## Reproduction and artifacts

`calibrate_tracking.py` now distinguishes estimated and measured distance labels.
Example using the saved evaluation:

```bash
.venv/bin/python calibrate_tracking.py \
  /tmp/drone-tracking-eval/20261011222411553-current-full-baseline.json \
  --start 0 --end 5 --distance-feet 12 --distance-kind estimated
```

Temporary artifacts in `/tmp/drone-tracking-eval/`:

- `distance-calibration-review.mp4`: full recording with current controller
  decisions and box measurements; no flight connection, no audio.
- `distance-far-reference.json` and `distance-near-reference.json`: measurement
  summaries with explicitly estimated distance labels.
- `20261011222411553-current-full-baseline.json`: all-frame tracking observations.

Evaluation used current `vision.py` (verified byte-identical to the evaluation
snapshot), YOLO11n, BoT-SORT, CPU, size 640, preprocessing on and video timestamps.
The first sole high-confidence person was selected once; no repeated selection.
No asynchronous processing or transport delay was simulated. No flight settings,
controller behavior, model weights or Target/Command/State formats were changed
in the initial measurement review above.

## Cropped-body fix and prototype

The subsequent investigation corrected the initial diagnosis: on the close-up
recording, appearance matching keeps the selected person through the approach.
Most of the late loss is because YOLO11n no longer produces a confident person
detection from the cropped torso. No appearance threshold was relaxed.

All five videos were rerun in full with pretrained YOLO11s, otherwise using the
same tracker, preprocessing, confirmation rules and initial-selection policy.
No model was trained on these clips. The exact initial boxes can differ between
detectors, so these are end-to-end replay comparisons, not controlled detector
accuracy measurements.

| Recording | YOLO11n confirmed | YOLO11s confirmed | Frames |
| --- | ---: | ---: | ---: |
| Short indoor | 20 | 28 | 540 |
| Overhead outdoor | 1369 | 1377 | 2087 |
| Outdoor walk/trailer | 1282 | 1048 | 2169 |
| Doorway/trash-can/two people | 1112 | 1172 | 2421 |
| Approximate distance reference | 557 | 689 | 692 |

YOLO11s keeps the close-up selected for every frame after the initial three-frame
confirmation. It has a regression on the outdoor walk, so **YOLO11n remains the
default**; YOLO11s is available explicitly via the existing `--model` option.
The girlfriend-alone interval (67–83 s) still has zero confirmed targets with
YOLO11s. The returning user is confirmed in 162 frames after 84 s; six sampled
frames spanning that interval were visually checked and match the user. This is
not a manual identity label for every frame.

The close-up YOLO11s run took median 60.8 ms, p95 70.5 ms per frame on CPU.
No frame in any of the five runs exceeded 300 ms. These batch measurements do
not establish live end-to-end latency.

### Application correction

`Controller.person_range_reason()` now explains cropped bodies as unknown range.
When the existing size limit stops a cropped person, the state is `LOCKED`,
not `REACHED`, so the UI/voice no longer claims that person has reached a known
distance. `main.py` displays the hold reason. The same neutral command and
follow-arming blocks remain. Full-body size stops still return `REACHED`.

Replaying all 7909 original detections through the old and new controllers in
both person and follow modes produced **15818 identical command comparisons**;
5676 status results changed from reached to locked. No new movement was enabled.

### Offline torso prototype

`body_range.py` and `preview_body_range.py` measure shoulder-to-hip height using
a pretrained YOLO11n pose model. Feet are not required. Both shoulders and hips
must be inside the frame with confidence >= 0.8, with simple posture checks.
Pose detections are associated with the already-selected person by box overlap;
weak/ambiguous associations produce no measurement.

The distance recording has 435 frames with a usable torso measurement. The first
five seconds supply 122 samples, using the user's approximate 12-foot label.
The illustration estimates about 12 feet early and about 7 feet at 16 seconds;
there is no independently measured distance validating those estimates. It
reports unknown around 18 seconds onward when the torso landmarks become weak
or cropped. It never carries a stale estimate forward.

Shoulder width alone changed markedly when the user turned while standing near
the door, which is why this prototype uses torso height. Torso height can still
vary with bending, perspective, camera tilt and person proportions. This is a
same-person experimental size cue, **not a calibrated flight distance**. It is
not connected to the controller, and the 3-foot endpoint remains an estimate,
not a permitted flight gap.

Full before/after video: `/tmp/drone-tracking-eval/cropped-body-before-after.mp4`.
The adjacent JSON records per-frame landmarks, measurements and missing values.
Full detection results use the `-stronger-full-baseline.json` suffix in that
directory. No raw videos or model weights were added to Git.

Validation: pipeline passed; 87 unit tests passed, covering existing tracking
and flight safety plus crop-status and torso-measurement regressions. Preview
frame count (692) and samples at 2, 16 and 22 seconds were visually checked.

## Airborne recording check

The full overhead outdoor recording (`20261011165245073.MOV`, 2087 frames,
83.9 seconds) was subsequently run through the torso prototype, using the saved
YOLO11s selected-person results. This recording supplies the aerial viewpoint;
no additional aerial footage was required for this check.

- 1377 frames have a confirmed selected person in that detector run.
- 397 frames pass the torso landmark checks (19% of the entire clip, 29% of
  confirmed-target frames). These are candidate measurements, not verified
  distance accuracy.
- 152 of those frames have at least one ankle that fails the confidence/edge
  checks. This shows that a torso candidate does not require reliable ankle
  detection; it does not prove that each such ankle is physically cropped.
- Some sampled side-on torso estimates remain ambiguous despite high confidence.
  Missing landmarks, camera angle and posture prevent continuous reliable range.

`preview_body_range.py` now accepts recordings without `--reference-feet`.
The airborne report has no reference distance and every distance value is null.
No indoor distance label was transferred to aerial footage, and no flight
control changes were made for this check.

Full review: `/tmp/drone-tracking-eval/airborne-torso-review.mp4`, with adjacent
JSON. The 2087-frame output and sampled overlays were checked. Pipeline and four
torso-measurement tests passed. This checks landmark availability from above;
it does not validate automatic approach from above.

## Combined size and ground experiment (2026-10-11)

User clarified that all five clips use the same drone camera, manually repositioned.
The reference camera was on a paper towel roll on a table, roughly 4 ft 5 in to
5 ft high, slightly tilted down by an unknown angle. The user's height is 5 ft
9 in. The airborne camera pointed almost fully downward. Advertised FOV is 60
degrees, with its axis unknown. These details do not supply calibrated intrinsics,
a ground plane, or per-frame airborne camera height and attitude.

`distance_combo.py` implements reference-size inverse scaling and calibrated
ray/ground-plane intersection. Ground depth is projected up the person's vertical
line to the same body anchor as the size cue, avoiding comparison of foot depth
with face depth in a tilted image. Unknown/expired camera pose, missing ankles,
near-horizon rays and incompatible geometry are rejected. Disagreeing size cues
or ground checks return unknown. The 30% rejection threshold is experimental,
not a measured error bound. Nothing imports this module into flight control.

The implementation uses the size principle from
[Asadullah's project](https://github.com/Asadullah-Dal17/Distance_measurement_using_single_camera)
and the calibrated ground-ray principle from
[Vehicle-Distance-Forensics](https://github.com/chennnlee/Vehicle-Distance-Forensics).
It does not port or run the latter project's SHARP reconstruction/RANSAC pipeline.
Face detection uses [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
because this environment's OpenCV 5 build has no Haar CascadeClassifier API.
Face detections must match the selected person's reliable nose/eye landmarks and
stay inside the image. Pose confidence is still heuristic, not visibility truth.

`evaluate_distance_combo.py` evaluated every frame of the reference recording and
every fifth frame of the other four. Both comparison panels use the same saved
YOLO11s tracking results; only the distance method differs. These are candidate
feature counts, not tracking accuracy or distance accuracy scores:

| Recording | Frames sampled | Torso usable | Face usable | Foot-contact candidates | Combined estimates |
|---|---:|---:|---:|---:|---:|
| 12 ft / 3 ft, 20261011222411553 | 692 | 435 | 63 | 0 | 435 |
| Airborne, 20261011165245073 | 418 | 79 | 21 | 20 | 79 |
| Outdoor, 20261011165024895 | 434 | 184 | 51 | 36 | 184 |
| Short indoor, 20261011164938939 | 108 | 0 | 0 | 0 | 0 |
| Trash-can, 20261011182901040 | 485 | 93 | 45 | 42 | 93 |

The first five seconds of the reference supply 122 torso samples but only TWO
face samples. The minimum reference count is ten, so face calibration is rejected.
No real clip has a calibrated ground plane; all ground distance outputs are null.
Consequently, the combined method falls back to the existing torso estimate and
adds **zero distance coverage** in this test. No claim of improved distance accuracy
is supported. Estimates transferred to other camera angles are explicitly marked
unverified; changing posture, view angle and subject proportions can invalidate them.

The approximate 3-foot interval at 20–25 seconds is held out of calibration.
All 124 frames there return unknown: the head and lower body are outside the image.
This is a useful failure case, not a validated 3-foot stopping distance. The short
indoor clip also inherits the poor initial selection of its saved tracking run;
its result does not measure performance after a better manual selection.

Artifacts: `/tmp/drone-tracking-eval/distance-combo/` contains five before/after
MP4s, per-frame/sample JSON, and `summary.json`. The reference comparison is
`20261011222411553-before-after.mp4` (692 frames, full duration). The other previews
play sampled frames at one-fifth the original fps to preserve approximate duration.
The original video image is fully visible in each panel; no doorway is masked.

Validation: all 104 discovered unit tests and `test_pipeline.py` passed, including
six combined-distance tests for tilted/level geometry, missing calibration,
conflicting estimates, target loss and cropped ankles. These test mathematical
behavior, not real-world flight accuracy. Models remain pretrained; no training
or flight-control integration occurred in this experiment.

## Offline horizontal centering (2026-10-12)

The user requested development with the existing recordings while the drone is
unavailable. `centering_preview.py` now separates the desired horizontal turn
from the command permitted by existing stop rules. The horizontal box center
does not require visible feet. Controls start off; loss, low confidence, invalid
coordinates and stale video clear the request immediately. Every translational
axis stays zero. Yaw starts beyond a 6% horizontal error, stops within 3%, and
ramps proportionally above that inner band, capped at 15% stick or the configured
lower limit. This hysteresis reduces toggling in synthetic boundary tests; it
does not establish real drone stability, latency tolerance or tuned gains.

The image-size stop remains enforced even when it prevents a desired turn.
`main.py` and the live controller are not connected to this prototype. No existing
flight behavior, arming policy, live-follow restriction or emergency control was
changed. The known nearly downward-facing aerial recording is excluded from the
forward-facing yaw assumption and produces neutral requests/output. Rotating the
drone there can rotate the image rather than move a person toward its center.

`preview_centering.py` replays all 7909 frames of the five saved YOLO11s tracking
runs and renders before/after videos. These reuse detections; they do not rerun
the currently edited tracker or simulate the camera moving in response to yaw.
The indoor distance clip now has 560 frames with a centering request (281 right,
279 left), versus zero turn outputs in the old controller. All 560 requests are
still withheld by the size stop; these are diagnostic direction decisions, not
newly permitted flight. In the trash-can clip, the prototype requests a turn in
932 frames and permits one in 496, versus 472 in the old controller. More turn
requests do not demonstrate better physical centering.

Artifacts: `/tmp/drone-tracking-eval/centering-review/`, containing five full-length
`*-centering-before-after.mp4` files, per-frame JSON, summary, and test output.
Both panels retain the original image and show the selected target; the new
panel distinguishes `Request` from `Allowed output` and explains holds.

Validation: 110 unit tests and `test_pipeline.py` passed. Six prototype tests
cover side-to-side turn direction, center-band hysteresis, immediate resets,
close/cropped-target holds, lower caps, and the unsupported downward view.
Replay assertions check turn sign, yaw limit, zero translation and enforced
loss/size stops. This is an offline candidate for further testing, not a claim
that automatic centering has been demonstrated in flight.

## iPhone-measured 19 / 10 / 3-foot recording (2026-10-12)

New source: `/Users/saltysalt/Downloads/20261012165212471.MOV`, 911 frames,
36.56 seconds at 24.92 fps, 1920x1080. User reports iPhone Measure readings of
19 feet at the opening station, 10 feet around 18 seconds, and 3 feet around
30 seconds, turning around at each station before a final approach to shut down
the camera. The drone was approximately 23 inches above the ground, camera facing
straight ahead. These are user-reported phone measurements and orientation, not
independently verified tape measurements or calibrated camera extrinsics.

The current YOLO11s selected-person tracker was run on every frame. It confirms
the selected person on 664/911 frames. Those counts are availability, not an
identity accuracy score. Pose association supplies usable torso measurements on
607 frames; the strict face and ground-contact checks supply zero. The feet remain
outside the usable view. Reported height/orientation alone cannot produce a valid
ground distance without usable ground contact and camera intrinsics.

The torso reference is fitted ONLY to the first stable 19-foot interval (0.5–3s):
62 samples, median normalized shoulder-to-hip height **0.2294620350**.
The calculation is `feet = 19 * 0.2294620350 / current_torso_height`.
The 10-foot and 3-foot samples are not used for fitting.

| Check | Interval | Estimates / frames | Median estimate | Middle 80% of estimates |
|---|---|---:|---:|---|
| 19 ft reference used for fitting | 0.5–3s | 62 / 62 | 19.0 ft | 19.0–19.5 ft |
| Continued turn at 19 ft | 3–13s | 210 / 249 | 18.6 ft | 18.0–19.4 ft |
| Held-out 10 ft station and turn | 17.5–25.5s | 192 / 199 | **9.84 ft** | **9.30–10.30 ft** |
| Held-out 3 ft station | 29.5–31.5s | 0 / 49 | Unknown | Unknown |

The 10-foot samples average 0.33 feet absolute difference from the reported phone
reading. This is agreement within one recording, with correlated frames and a
phone reference; it is not a general distance-accuracy guarantee. The 19-foot
fit interval reproducing 19 feet is calibration, not independent validation.
All selected-person confirmations are lost in the chosen 3-foot interval, and
the needed torso landmarks are cropped. No 3-foot estimate is fabricated from
the label or carried forward from the previous position. Later approach frames
are excluded from calibration/validation distance labels.

At the 19-foot reference, the box is about 17% of the image width and 67% of its
height. The current 65%-height stop therefore holds even at this far reference.
At 10 feet it is about 30% wide and 99% tall. This demonstrates why image-size
stops are not calibrated distance stops; it does not authorize relaxing them.
Existing close-stop, lost-target, arming and flight behavior remain unchanged.

Permanent reference metadata and interval reports:
`calibration_data/20261012165212471.json` (offline experiment only).
Full analysis and comparison video:
`/tmp/drone-tracking-eval/new-distance/distance-analysis.json` and
`/tmp/drone-tracking-eval/new-distance/measured-distance-before-after.mp4`.
The comparison shows the current image-size hold beside the experimental torso
estimate, with user distance labels only during the marked comparison intervals.

### Applying the new scale to the five earlier recordings

The stored torso measurements from all five earlier comparisons were reviewed
against the new 19-foot reference. This reuses 692 full-frame measurements for
the earlier indoor distance clip and every-fifth-frame measurements for the
other four. The nearly downward-facing aerial clip is excluded from transfer.
At the initial transfer comparison, other camera angles had no measured check.
The user subsequently supplied the indoor 14-foot measurement described below.
All `verified_distance_feet` fields remain null because phone reference readings
are not independently surveyed distances. Original evaluations and live settings
are unchanged; the transfer review separately records the new reference check.

Changing the reference from `12 * 0.3089571074` to `19 * 0.2294620350` multiplies
each available torso estimate by **1.17594** (about 18% larger). It does not add
usable landmarks, repair lost identity matches, compensate for camera angle,
or establish better accuracy on old footage. The previous indoor reference
at 0–5 seconds changes from 12.0 to 14.11 feet in this transfer trial; at
15.5–16.5 seconds it changes from 6.93 to 8.15 feet. The 20–25-second close-up
still supplies no estimate. The original 12-foot label was approximate; it was
subsequently corrected to a user-reported 14-foot iPhone measurement below.

Transferred candidate counts are 435/692 for that indoor clip, 93/485 for the
trash-can clip and 184/434 for the outdoor walking clip. The short backlit indoor
clip remains 0/108. The airborne clip's 79 torso candidates are not converted
using this new reference because of its nearly downward view. Counts measure
availability only. The indoor opening now has one measured-reference comparison;
the other clips and indoor close-up still lack successful measured checks.

Audit summary: `calibration_data/reference-transfer-20261012.json`. Per-sample
trial results: `/tmp/drone-tracking-eval/reference-transfer-review/`.
The evidence now includes a measured reference, a separate 10-foot check in the
new fixed-camera recording, and the indoor 14-foot comparison below. This is not
a trained distance model or a universal calibration for every camera angle.

### Indoor doorway measurement corrected to 14 feet

The user subsequently reported measuring the starting doorway position in
`20261011222411553.MOV` with iPhone Measure and obtaining **14 feet**, correcting
the earlier approximate 12-foot label. The 19-foot-fitted calibration had already
predicted a median **14.11 feet** for this 0–5-second interval before that
measurement was supplied. The median prediction differs from the phone reading
by about **0.11 feet (1.3 inches)**. This is agreement at one station, not a claim
of 1.3-inch general accuracy.

The existing calibration was not refitted to make this result match. This is a
useful check in a second recording and supports the new scale at the indoor
doorway, alongside the 9.84-foot result at the new clip's 10-foot station. It does
not resolve missing landmarks at 3 feet or validate the airborne camera angle.
The older 12-foot reference remains documented as the historical assumption;
future reference checks for the indoor starting position use the corrected
14-foot reading.

The measurement, prediction distribution and difference statistics are saved in
`calibration_data/20261011222411553-validation.json` and included in the updated
transfer review summary. No model training, flight thresholds or controls changed.

## Grounded measurement screen and recording

`main.py --distance-check --model yolo11s.pt --device cpu` now launches the
camera-only `distance_check.py` screen. It loads the existing 19-foot torso
reference, associates pose and selected-person detections on the same frame,
and displays feet or unknown. Missing/ambiguous torso features, unconfirmed
selection, nonmatching aspect ratio and results older than 300ms produce unknown;
no old distance is retained after a failure. Model warm-up precedes video startup.

R saves labeled and raw video plus a per-frame JSONL journal; M marks a phone
measurement for later annotation. The journal embeds the calibration and actual
session/receive timestamps, while the visible session clock is recorded in the
HUD. Fixed-15fps silent MP4 playback is not a wall-clock timing measurement.
Recorded markers initially have null measured distances until the user supplies
the corresponding readings. This collects validation data without changing scale.

The mode forces flight control off and never constructs DroneLink. It is for a
grounded drone and has no flight hotkeys. Existing flight modes and their stop,
arming and landing behavior are unchanged. New recordings are ignored by Git.

Validation: 118 unit tests and pipeline passed, including eight new distance-screen
tests for measured-reference scaling, stale/lost results, ambiguous/cropped pose,
aspect-ratio mismatch, recording contents/timestamps, writer cleanup, and camera-only
R/M operation with no flight link. A real-model replay around 18 seconds of the new
recording produced roughly 9.3–9.7 feet and successfully recorded a 9.66-foot marker.
The rendered screen was visually inspected. Live drone-stream operation is not yet
tested because no hardware connection was available during implementation.

### First user distance-check recording: marker timing clarification

The user supplied `distance_recordings/distance_1791839821003176000.jsonl`
and reported Marker 1 as **17 feet** and Marker 2 as **about 10 feet 2 inches**.
They confirmed standing at each measured spot **before walking to the Mac to
press M**. The button-press frames therefore are not confirmed distance references.
The original journal is preserved; user labels and timing qualifications are in
`calibration_data/live-distance-1791839821003176000-annotations.json`.

Recording starts at session 57.57s, only 1.38s before Marker 1. The opening shows
the subject reaching toward the car, so the earlier 17-foot station may not be
recorded. No validated 17-foot comparison is assigned. A candidate standing
section before Marker 2, session 92–94.5s, has estimates in 32 of 33 frames:
10th percentile / median / 90th percentile **10.15 / 10.51 / 10.83 feet**.
Visual movement, rather than numerical agreement, identifies this candidate;
its correspondence to the measured station remains unconfirmed. It is not used
for fitting or a general accuracy claim. Session clock values differ from MP4
playback time. No distance scale or flight controls changed.
