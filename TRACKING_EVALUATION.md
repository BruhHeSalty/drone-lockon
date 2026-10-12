# Recorded-video tracking check — October 11, 2026

The selected-person tracker compared every detection with a single appearance
captured at the initial click. In these recordings, pose changes and partial
body views often failed that comparison even when the detector was confident.

`vision.py` now retains that original reference and a slowly updated recent
reference. It updates the recent reference only after the existing confirmation
and position checks pass, on the same uninterrupted track ID, at confidence
>= 0.80, with only one selectable person. Missing, weak, ambiguous or jumping
detections do not teach it; the first confirmation after reacquisition does not
teach it either. Reselection resets both references.

The confidence requirement for a confirmed output remains 0.60, with three
consecutive observations. Lost/uncertain observations still return `found=False`.
No flight control, Target format, deadline or model weights changed.

## Replay results

All 4,796 frames were evaluated in order using YOLO11n, BoT-SORT, 640-pixel
inference, CPU, and drone image cleanup enabled. Video timestamps drove the
continuity checks. Before/after selection logic received the same cached
detections and appearance features. Replaying the original code reproduced its
baseline counts exactly for all three clips.

The first scenario selects the first sole detection at confidence >= 0.60 and
does not click again. Files are in the user's Downloads folder.

| Video | Frames | Selection time | Strong detection frames | Confirmed before | Confirmed after |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20261011165245073.MOV` — overhead outdoor | 2,087 | 0.00 s | 1,522 | 226 | 1,096 |
| `20261011165024895.MOV` — outdoor | 2,169 | 2.41 s | 1,358 | 4 | 1,010 |
| `20261011164938939.MOV` — indoor backlighting | 540 | 0.00 s | 425 | 20 | 20 |

A second scenario checks sensitivity to clicking later, when a clearer body
view is available. Each run again makes only one selection.

| Video | Selection at/after | Confirmed before | Confirmed after |
| --- | ---: | ---: | ---: |
| Overhead outdoor | 13.0 s | 51 | 1,134 |
| Outdoor | 8.0 s | 461 | 878 |
| Indoor | 7.5 s | 305 | 350 |

The two outdoor clips informed the change. The indoor clip was checked after
the change was fixed; it was not used to tune the update rate or thresholds.
Its early-selection failure remains: an abrupt appearance change after a gap
can still need an explicit new click on a clear person box.

These are confirmed-frame counts, **not labeled tracking accuracy**. Counts
include off-screen periods, depend on selection timing, and do not establish
identity accuracy with multiple people. Replays do not simulate live frame
dropping, transport delays or flight. Sampled newly confirmed outdoor boxes
were visually inspected. The model was not trained on these videos.

## Review and validation

- `/tmp/drone-tracking-eval/tracking-before-after.mp4`: 24-second comparison
  from both outdoor clips. Green is confirmed; gray is a detection without a
  confirmed selection. It uses the first-selection scenario above.
- `/tmp/drone-tracking-eval/`: original-code snapshot, evaluation scripts,
  cached observations and per-frame JSON results. Temporary files can be removed
  by the operating system.
- `.venv/bin/python test_pipeline.py`: passed.
- `.venv/bin/python -m unittest test_person_tracking`: 22 passed, including six
  new appearance-memory regressions.
- `.venv/bin/python -m unittest test_flight_safety test_person_follow`: 50 passed.
- `git diff --check`: passed.

To review the overhead recording interactively with the updated tracker:

```sh
DRONE_VISION_PREPROCESS=on .venv/bin/python main.py --person --dry-run --video /Users/saltysalt/Downloads/20261011165245073.MOV
```

Click a clear person box to select. This command replays the recording without
sending flight commands.

## New clip: departure, return and two people

`20261011182901040.MOV` is 97.375 seconds, 1,920 × 1,080, with 2,421 frames.
This is a check of the updated tracker above, with no further tracker tuning.
The evaluation makes one simulated click on the sole strong person detection
at/after 6 seconds, when the intended subject is clearly visible indoors.
It uses the same model, preprocessing and video-time replay method above.

| Source interval | Observed scene | Confirmed output |
| --- | --- | --- |
| 35–55 s | Walk to the trash can and back | 448 / 497 frames (90.1%) |
| 67–83 s | Selected subject absent; girlfriend enters and remains alone | 0 frames |
| 84 s–end | Subject returns; both people visible | 138 / 332 frames |

The first confirmed output after returning is at **84.384 seconds**, on track
176. Sampled confirmed boxes after the return were visually checked and are on
the intended subject, not his girlfriend. The zero outputs during 67–83 seconds
cover every replayed frame in that interval. This checks the lost-target output;
no physical hover or flight response was tested.

The test also exposes remaining dropouts: only 15 / 373 frames were confirmed
during 20–35 seconds, when the subject passes the doorway and moves outdoors.
Appearance mismatch is the main rejection reason there. Confirmation also
drops during the close interaction between the two people, with both appearance
and overlap/uncertainty checks contributing. A higher total confirmation count
alone would not prove improvement on this clip: selecting the other person would
be a regression.

Review video: `/tmp/drone-tracking-eval/new-video-tracking-review.mp4`.
It includes source intervals 35–55 and 63–97.375 seconds, with original source
timestamps in the overlay. Green means confirmed selection; gray means a person
detection only. No audio is included. Supporting JSON, cached detections and
the current-code snapshot are in the same temporary directory.

The report update was followed by a passing `.venv/bin/python test_pipeline.py`.

### Full review and direct comparison with the original tracker

The full annotated review is
`/tmp/drone-tracking-eval/new-video-full-tracking-review.mp4`. It contains all
2,421 source frames in order, including the doorway: 97.375 seconds, with no
sections removed. The overlay is silent and uses the updated tracker results.

Replaying the pre-improvement tracker (`vision_before.py`) against the same
cached observations and the same selection at/after 6 seconds produced:

| Interval | Original confirmed frames | Updated confirmed frames | Frames evaluated |
| --- | ---: | ---: | ---: |
| Entire clip | 27 | 743 | 2,421 |
| Doorway / outdoor transition, 20–35 s | 0 | 15 | 373 |
| Trash-can walk, 35–55 s | 0 | 448 | 497 |
| Selected person absent, 67–83 s | 0 | 0 | 398 |
| Return / both people, 84 s–end | 0 | 138 | 332 |

At the source frame rate, 27 and 743 confirmed frames represent approximately
1.1 and 29.9 seconds of accumulated confirmation, respectively, not continuous
lock durations. These are single-selection replay results, not a general
accuracy percentage. In particular, the doorway remains poor: 15 / 373 frames
(4.0%) confirmed. Uploading footage supplied test evidence for the implemented
appearance-memory change; the detector model has not been retrained.

## Indoor appearance update

The next change addresses abrupt jumps in the appearance score when indoor
color noise or exposure changes move pixels between histogram bins. Hue,
saturation and relative-brightness histograms now share weight with neighboring
bins. Hue wraps around red; saturation and brightness use bounded edges.
Saturation and the upper/lower body layout are retained. The detector, image
preprocessing, matching threshold, confidence requirement and three-frame
confirmation rule are unchanged.

Comparison below is against the previous appearance-memory improvement, using
the same one-time selection at/after 6 seconds in `20261011182901040.MOV`:

| Source interval | Before this update | After this update | Frames |
| --- | ---: | ---: | ---: |
| Indoors, 6–20 s | 70 (20.1%) | 269 (77.3%) | 348 |
| Doorway / transition, 20–35 s | 15 (4.0%) | 102 (27.3%) | 373 |
| Trash-can walk, 35–55 s | 448 | 452 | 497 |
| Selected subject absent, 67–83 s | 0 | 0 | 398 |
| Return / both people, 84 s–end | 138 | 155 | 332 |
| Entire clip | 743 | 1,112 | 2,421 |

All confirmed outputs in the return interval use the intended subject's track
176, with no confirmed outputs on his girlfriend's track 142. Sampled recovered
boxes in the indoor/doorway section were visually checked against the video.
A candidate that discarded saturation information was rejected because it
incorrectly selected the girlfriend during the subject's absence.

Regression replays on the earlier clips, using their original selection times:

| Clip | Before this update | After this update |
| --- | ---: | ---: |
| Overhead outdoor | 1,096 | 1,369 |
| Other outdoor | 1,010 | 1,282 |
| Short backlit indoor, selection at start | 20 | 20 |
| Short backlit indoor, selection at/after 7.5 s | 350 | 350 |

The short indoor clip's early-selection failure remains unresolved, as does
much of the darkest doorway passage. These are confirmed-frame counts from
recorded replays, not labeled accuracy or live-flight results. All four clips
have now been used for development or regression checks; further footage would
provide an independent check of generalization.

Features were recomputed from every source frame with the final implementation;
cached detector outputs were reused because detection did not change. The full
97.375-second, 2,421-frame comparison is
`/tmp/drone-tracking-eval/indoor-full-before-after.mp4`, with no source sections
removed and no audio. The left panel is the previous version, and the right
panel includes this indoor update.

Validation: 26 person-tracking tests (including four new appearance/lighting
tests), 50 flight-safety/follow tests, and `test_pipeline.py` passed.

## Fresh full-video test and selecting another person

The current implementation was run again on **every frame of all four source
videos**, including fresh YOLO/BoT-SORT inference and appearance extraction:
7,217 frames in total. Code snapshot:
`/tmp/drone-tracking-eval/vision_all_videos_snapshot.py`. No tracker settings or
production code were changed during this test.

| Video | One-time selection | Confirmed frames / total | Median / p95 processing ms |
| --- | ---: | ---: | ---: |
| Short indoor `20261011164938939.MOV` | 0.000 s | 20 / 540 | 39.7 / 46.4 |
| Overhead outdoor `20261011165245073.MOV` | 0.000 s | 1,369 / 2,087 | 41.3 / 53.8 |
| Outdoor `20261011165024895.MOV` | 2.413 s | 1,282 / 2,169 | 39.7 / 44.5 |
| Latest `20261011182901040.MOV` | 6.033 s | 1,112 / 2,421 | 40.8 / 47.1 |

These exactly reproduce the prior cached-detection replay totals. Denominators
include frames before selection and when the intended person is absent; these
are not accuracy percentages. Timing covers synchronous `find()` calls in this
offline batch, not live transport, rendering or frame scheduling. No call took
300 ms or more. Every missing-target result was checked through the person-mode
controller and returned all-zero commands. No flight was performed.

Selecting the clearer view at 7.508 seconds in the short indoor clip instead
gave **350 / 353 remaining frames confirmed**. This does not resolve the failure
after the original early selection; it shows how much the starting appearance
matters.

In the latest clip with the original subject selected, the full fresh run again
produced zero confirmed targets throughout 67–83 seconds (398 frames). From
84 seconds onward all 155 confirmed outputs used his track 176, first at
84.142 seconds.

A separate replay using the fresh detections selected the girlfriend at/after
74 seconds. It produced **242 / 249 confirmed frames during 74–84 seconds**,
and **128 / 332 from 84 seconds to the end**, all on her track 142. It never
confirmed his track 176 in that scenario. Sampled confirmed boxes were visually
checked. Confirmation still pauses during overlap and uncertain matches.

The person detector is generic: click the intended person's current box to
select them, or click another person to replace the selection. It detects
multiple people but produces one selected Target. Current person mode filters
YOLO to the `person` class; pets, cars and other object categories are not
enabled in this mode. These recordings test two individuals, not every possible
person, outfit or environment.

Full-length comparison showing either selection:
`/tmp/drone-tracking-eval/choose-either-person-full.mp4` (97.375 seconds, all
2,421 frames, silent). Left selects the original subject at 6 seconds; right
selects the girlfriend at 74 seconds. Fresh inference reproduced the cached
selection-comparison rows exactly. Summary JSON:
`/tmp/drone-tracking-eval/all-videos-current-summary.json`.
