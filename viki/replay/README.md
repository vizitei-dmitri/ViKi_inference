# viki.replay — trajectory execution

**Stage 5** · `plan.h5` → `replay.h5` · paper §3.8

Execute the synthesised joint trajectory and record measured arm states.
UR10e arm-only execution is implemented using `ur-rtde`; it has not yet been
validated on a physical robot. Hardware grippers, collision/singularity replay
classification and the feedback re-solve loop remain unimplemented (§3.8).

## Files

| file | status | what |
|---|---|---|
| `driver.py` | dryrun working, UR3 stub | `RobotDriver`, `ProprioLog`, lazy driver loading |
| `plan.py` | working | full approach + demo validation, joint-order/model gate, linear resampling and discrete velocity/acceleration bounds |
| `ur.py` | implemented, hardware validation pending | read-only inspection, UR10e identity/remote/start gates, watchdog, servoJ, measured telemetry, cleanup on error |
| `screen.py` | partial | finite telemetry, registered physical joint limits and measured tracking residual; unavailable checks cannot silently yield `pass` |
| `run.py` | working | immutable execution logs, source-frame summary, partial failure artifact; nonzero `max_resolves` explicitly raises |
| `manager.py` | working | server cancellation tokens; cancellation never calls RTDE from a second thread |
| `simulation.py` | working | offline MuJoCo position-actuator/kinematic preview, model asset hashes, paused GUI, MP4 and model-scoped reports |
| `live.py` | implemented, hardware validation pending | spawned arm owner, relative bind/clutch, freshness gates, online IK, MuJoCo/RTDE backends and independent durable log process |

See [the UR10e setup and first-run guide](../../docs/ur10e_execution.md).

```bash
docker compose build ur
docker compose run --rm ur replay <episode> --check --arm-only
docker compose run --rm ur replay <episode> --inspect --host <robot-ip> --arm-only
docker compose run --rm ur replay <episode> --driver ur10e --host <robot-ip> --arm-only
```

`--check` never connects. `--inspect` only reads Dashboard/RTDE and does not
upload a control script. Execution requires a UR10e plan with canonical joint
order, converged IK and finite nonnegative floor/collision/joint/velocity plan
margins. It never moves to home, wraps joint angles, powers on, releases brakes,
unlocks a protective stop, or retries motion automatically. Put the stationary
arm at the checked `start_q_rad` on the pendant before replay. Default speed
scale is 0.1 at 125 Hz, with 0.25 rad/s and 0.5 rad/s² discrete command bounds.
Installed controller safety checks cover archived waypoints; neither these nor
the planner metrics validate the complete path against the real environment.

## Contract

- **out:** `replay.h5` — keys in `contracts.REPLAY_KEYS`:
  `q_attained (T,nq)`, `gripper_attained (T,)` as normalised opening,
  `controller_residual (T,)`
  *(NaN under dry-run)*, `verdict` (`pass`｜`reject`｜`dry-run`｜`arm-only`｜`test-stop`),
  `rejection_cause`, `resolve_attempts`, `robot`, `dt`, `source_dt`, `driver`,
  `hardware`, `completed`, `gripper_measured`, `plan_sha256`, `execution_log`.

The summary omits the approach to retain the source-demo frame correspondence;
its `dt` is the slowed execution time step. Arm-only gripper measurements are
NaN, with `gripper_measured=false`, and its successful verdict is `arm-only`.
Dryrun remains explicitly synthetic. Neither qualifies for LeRobot training.

`--stop-after 10` limits a hardware bring-up trial to ten seconds of command
streaming. A normal timed stop records `test-stop`, `completed=false` and
`test_duration_limit`; it does not mark the whole episode replayed. Each new
attempt starts at the original checked pose, including after a timed stop.

Each attempt keeps `intermediates/replay/<run-id>/execution.h5`, including the
approach, source commands, provenance and partial data on a handled abort.
Hardware logs additionally have control-rate `q_commanded`, `q_observed` before
the command, `q_attained`/`qd_attained` after the period, active-controller
`tcp_pose_robot` (metres/rotation-vector radians), safety mode, host monotonic
command time and controller timestamp. The residual is maximum absolute joint
tracking error in radians, not an internal controller diagnostic.

The active UR TCP depends on the pendant installation and is distinct from
ViKi's dynamic gripper contact midpoint. RTDE getters use the receive cache;
the recorded timestamp is a controller clock, not a camera timestamp. Summary
rows take the first control sample at/after each source-frame time. Keep the
control-rate log for subsequent policy work.

## Remaining work

For causal human following use `viki live`, documented in
[live/README.md](../live/README.md) and the [live guide](../../docs/live_human_run.md).
Its default executor is 50 Hz, separate from the offline replay's 125 Hz.
Binding uses the current measured arm state, and physical control is uploaded
only on an explicit bind. Invalid/stale inputs clear binding and brake; input
recovery requires another bind. Each session preserves accepted commands,
measured joints and reasons even when stopped or faulted.

Validate staged motion on hardware; register the physical robot base and tool;
add an independently connected measured gripper/hand driver and synchronized
robot-execution camera capture. Implement collision/singularity classification
and the bounded re-solve edge separately. Camera data in `raw/` is a human
demonstration and must not be relabelled as synchronized robot observations.
