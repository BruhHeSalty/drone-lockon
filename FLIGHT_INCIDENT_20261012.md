# Unexpected takeoff on close — 2026-10-12

## Operator reports and scope

The operator reported a successful manual hover and landing, followed by a
person-mode attempt that did not initially lift off. They later clarified that
the drone took off when the program closed. A separate landing request worked;
the operator then removed the battery. These are operator observations, not
telemetry. No live tests or new protocol commands were issued during this
investigation after battery removal.

## Recorded sequence (UTC)

| Time | Recorded event |
| --- | --- |
| 21:41:21.941 | Manual test submitted takeoff. |
| 21:41:59.823 | Manual test submitted landing; operator reported successful hover/landing. |
| 21:42:53.799 | Person-mode test submitted takeoff; no physical acknowledgement exists. |
| 21:43:17–21:43:50 | Six further takeoff requests were rejected as already requested. |
| 21:45:02.409 | Person-mode shutdown submitted a landing-intent toggle. |
| 21:45:03.159 | Original control session ended. |
| 21:46:23.364 | A separate emergency connection submitted one landing-intent toggle. |

Source journals: `flight_logs/flight_1791841275751377000.jsonl`,
`flight_logs/flight_1791841366654378000.jsonl`, and
`flight_logs/flight_1791841583347851000.jsonl` (local ignored recordings).
Original journals were preserved. Adjacent operator-report JSON files record
the user confirmations separately.

## Confirmed software defect and remaining uncertainty

`DroneLink._tick()` sets `self.flying = True` when it prepares a takeoff request.
There is no aircraft acknowledgement or altitude/motor-state telemetry behind
that flag. `close()` calls `land()`, which trusts the flag. The installed
`WifiUavRcProtocolAdapter` encodes both intentions with action bit `0x01`;
its different TAKEOFF/LAND log labels do not change the action bytes.

If the first request does not produce takeoff, shutdown can send the takeoff
action to a grounded aircraft. The operator's observed sequence is consistent
with this exact failure path. The journals prove local submission, not drone
receipt or execution; why the first request had no visible effect is unresolved.
There is no evidence of a delayed takeoff request being retained until quit.

The current encoder exposes no separate, verified land-only command. Emergency
stop is a different action and cuts motors; it is not a substitute for landing.
Repeated toggle transmissions, assumed flight state, or speculative throttle
commands are not a safe repair. A software close callback also cannot guarantee
delivery if the application is forcibly killed or the connection is lost.

## Containment and intended workflow

- All normal live modes and direct live `DroneLink` construction now fail before
  control sockets or sender threads are created. No CLI bypass was added.
- Camera-only distance measurement and dry-run tracking remain available.
- Person-mode A activation now checks local takeoff/settling readiness. In the
  offline simulated sequence, A before T and A during settling do nothing;
  only T requests takeoff, and a later A enables yaw-only tracking. This local
  gate does not establish actual hover or resolve the toggle defect.
- Existing Target/Command/State formats, SPACE handling and movement limits
  are retained behind the live-control block. Shutdown behavior was subsequently
  changed to emergency motor stop, as described below. Automatic shutdown
  landing is **not** claimed fixed or physically validated.

The takeoff sequence remains: explicit T takeoff; operator observes stable
hover; explicit A enables yaw-only centering. The operator subsequently changed
the exit requirement to emergency motor stop regardless of assumed flight state.
Re-enabling live control still requires a verified
land-only mechanism or reliable flight-state feedback and corresponding failure
handling. Protocol investigation and any eventual physical verification must
start with propellers removed. Do not infer successful landing from sendto().

## Offline verification

Regression tests cover rejection of every live main mode before hardware/model
construction, rejection of direct live links before adapter/sender construction,
continued dry-run availability, and T-before-A activation with fake hardware.
The existing control/encoder tests use fake sockets and disabled sender threads;
only those simulations patch out the live interlock.

## Implemented shutdown repair: emergency motor stop

At the operator's explicit request, normal close/Q (Escape in manual mode) now
cuts motors instead of trying to land. This is an emergency action: an airborne
drone drops. SPACE remains the separate explicit landing request. Forced process
termination, a dead computer, or lost Wi-Fi cannot guarantee command delivery.

`DroneLink.close()` atomically marks the link closed, cancels pending takeoff,
landing, calibration, flip and movement, and queues the existing motor-stop
burst. It submits the first packet synchronously without waiting for the sender
thread, then keeps the shared socket open for the short flush before releasing
it. It never consults the assumed `flying` flag to choose an exit action. Stop
submission counts are recorded without claiming the physical motors stopped.
Repeated close calls do not enqueue a new burst. A failed send is reported.

Regression tests reproduce locally accepted but physically unconfirmed takeoff
followed by shutdown; queued takeoff; queued landing/calibration; no takeoff;
socket failure; and repeated close. The real installed encoder, with fake
sockets and no background thread, emits 40 stop packets with action byte `0x02`,
neutral axes and valid checksums, and no takeoff/landing action bit. Existing
exception cleanup still flushes control before releasing the video socket.

This fixes the observed shutdown toggle path in software. It does not establish
physical stop reliability or resolve SPACE's ambiguous landing toggle. The live
flight interlock remains in place; nothing was sent to the drone for these tests.
