# Changes

## 2026-10-12 — Restricted live tests and recorded distance view

- Recorded subsequent failed-lift/flip findings and the unresolved assumed-flight
  state limitation; this checkpoint does not claim a fix for physical instability.
- Expanded media/session-log exclusions and removed identifying local paths and
  personal height from current documentation. Prior Git history is unchanged.
- At the operator's request, re-enabled explicit `--flight-test` and yaw-only
  `--person`. Unrestricted control, `--manual` and live following remain blocked.
- Added final yaw-only enforcement to both control and video ACK packets: other
  axes stay neutral, yaw is capped at 0.15 or the lower configured limit, and
  invalid yaw produces neutral output. T and then a separate A remain required.
- Added `--person --distance-view`: experimental feet/UNKNOWN replaces the
  animated lock-on HUD. `flight_distance.py` runs pose analysis in a background
  worker using matched source frames; stale or changed selections discard old
  estimates. Distance does not feed flight control.
- T starts clean video, annotated video and a timestamped journal before takeoff.
  R toggles recording; M bookmarks it. Pose/recording failures are isolated from
  flight key handling. Measured stationary intervals are needed for later review.
- Added tests for scoped live-mode permissions, final command restrictions,
  asynchronous distance freshness/selection, recording output and disk failures.
  Shutdown still requests motor stop; the existing landing-toggle and physical
  acknowledgement limitations remain unresolved. Verification uses fake hardware.

## 2026-10-12 — Distance review, centering previews and shutdown repair

This update saves the work since `9ff5490` on `codex/person-tracking-review`.
This commit initially disabled live flight; the subsequent restricted resumption
is documented above. Camera-only distance measurement and offline previews remain
available. The tests below verify software behavior, not flight safety.

### Flight controls and the shutdown incident

- Added a live-flight interlock in `drone.py`, checked by `main.py` before
  starting models, video or control connections. Direct live `DroneLink`
  construction is blocked too; there is no CLI override.
- Fixed the shutdown path that could send a takeoff-capable landing toggle
  after a takeoff request with no confirmed physical response. Closing/Q now
  requests emergency **motor stop**, regardless of the assumed flight state.
  Escape does the same in manual mode. An airborne drone will drop.
- Shutdown atomically cancels pending takeoff, landing, calibration and movement,
  submits the first motor-stop packet immediately, then keeps the shared socket
  alive for the stop burst. It records local submission counts without claiming
  physical acknowledgement. Repeated close calls do not start another burst.
- Kept SPACE as the explicit landing request. The current protocol uses the
  same bit for takeoff and landing, so this remains unresolved for live use.
  Network failure or forced process termination can also prevent stopping.
- Added a local readiness check before A can enable person-mode tracking:
  T requests takeoff, then the operator observes hover before pressing A.
  Clicking a person does not launch or enable person-mode autopilot. The timer
  and readiness check do not measure actual airborne state.
- Updated HUD, console instructions and README for the changed exit action.

Full evidence, chronology and limits: [flight incident report](FLIGHT_INCIDENT_20261012.md).

### Person framing and cropped bodies

- `controller.py` now explains why an image-size stop or cropped body prevents
  movement. A cropped large box stays `LOCKED` with distance unknown instead
  of claiming `REACHED`. Existing size stops and speed limits remain in force.
- `centering_preview.py` computes horizontal turning wishes with a 6% start
  band and 3% stop band. It separately reports the permitted command, retains
  the image-size stop, rejects stale/unconfirmed targets, and never requests
  translation or altitude changes. The known downward camera view is unsupported.
- `preview_centering.py` renders before/after reviews from saved tracking data.
  This centering prototype is not connected to live control; video replay does
  not prove that an airborne drone can hold position or keep someone centered.
- Added explanatory comments through the existing tracking, HUD, voice,
  controller, configuration, benchmark and pipeline code. Appearance matching
  remains uncertain and voice range labels based on box size are not measurements.

### Distance calibration and comparison tools

- `calibrate_tracking.py` summarizes selected-person boxes and separates cropped
  versus fully visible samples and measured versus estimated distance labels.
- `body_range.py` measures shoulder-to-hip image height when the required
  landmarks are sufficiently confident and visible. Legs can be cropped;
  missing shoulders/hips, posture and camera changes still limit the estimate.
- `preview_body_range.py` and `evaluate_distance_combo.py` generate offline
  comparisons; `distance_combo.py` explores size and ground-plane estimates.
  Ground-based estimates require explicit camera/ground calibration and valid
  contact points. They do not invent a distance when those inputs are missing.
- Saved the user-reported iPhone Measure references and audit summaries under
  `calibration_data/`. The current torso reference uses the 19-foot station.
  The separate 10-foot interval produced a median 9.84-foot estimate; the older
  doorway interval produced 14.11 feet and was later labeled 14 feet by the user.
  These limited checks are not a claim of general accuracy. The cropped 3-foot
  interval had no usable estimate. No model weights were retrained.
- Preserved the first live measurement labels (17 feet and about 10 feet
  2 inches) separately from the source journal. The operator walked to the Mac
  before pressing M, so marker frames are not treated as distance ground truth.

Full measurements, assumptions and reproduction notes: [calibration review](CALIBRATION.md).

### Grounded measurement and recording

- Added `main.py --distance-check` through `distance_check.py`. It forces flight
  commands off and never constructs a flight-control link.
- The display uses the measured torso reference and shows `UNKNOWN` for stale,
  ambiguous, cropped or incompatible input. Pose and selected-person association
  use the same frame, and result age is checked again after inference.
- R records raw video, annotated video and a JSONL journal. M adds numbered
  bookmarks. Session timestamps preserve real timing; fixed-15fps playback is
  not a wall clock. Recordings are silent. The embedded calibration makes a
  saved result traceable to the reference used at the time.
- `distance_recordings/` joins raw flight recordings and `flight_logs/` in
  `.gitignore`. Source videos, generated review movies, local model caches and
  per-frame private recordings are not included in this Git snapshot. The
  review scripts require those local inputs; compact calibration summaries are
  included. No credentials are needed for the offline tests.

### Verification

```bash
.venv/bin/python -m unittest discover -p 'test_*.py'
.venv/bin/python test_pipeline.py
git diff --check
```

126 unit tests and the pipeline pass in the project environment. Added coverage
includes torso validation, distance fusion rejection, cropped-body status,
centering decisions, recording timestamps/cleanup, live-launch blocking,
T-before-A sequencing and emergency shutdown. Real-encoder shutdown tests use
fake sockets and verify the stop bit, neutral axes and checksums without sending
anything to a drone. All results remain offline verification; live operation
must not be inferred from a passing test suite.
