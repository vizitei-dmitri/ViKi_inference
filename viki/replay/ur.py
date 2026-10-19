"""UR10e arm-only execution via ur-rtde; no automatic home or fault reset."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import logging
from pathlib import Path
import re
import signal
import socket
import threading
import time

import numpy as np

from viki.replay.driver import ProprioLog, RobotDriver
from viki.replay.plan import URReplayOptions, servo_commands, validate_commands

logger = logging.getLogger(__name__)


@contextmanager
def _motion_signals():
    """CLI cleanup also works when a shell background child inherited SIG_IGN."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)
    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _dashboard_info(host: str) -> dict:
    """Read identity and remote mode without sending control commands."""
    with socket.create_connection((host, 29999), timeout=3) as sock:
        with sock.makefile("rb") as stream:
            if not stream.readline(4096):
                raise RuntimeError("Dashboard closed before its greeting")
            result = {}
            for key, command in (
                ("model", "get robot model"), ("version", "PolyscopeVersion"),
                ("serial", "get serial number"), ("remote", "is in remote control"),
            ):
                sock.sendall((command + "\n").encode("ascii"))
                response = stream.readline(4096)
                if not response or not response.endswith(b"\n"):
                    raise RuntimeError("invalid Dashboard reply")
                result[key] = response.decode("utf-8").strip()
    # Dashboard calls an e-Series UR10e 'UR10'; PolyScope 5 identifies e-Series.
    if result["model"].upper() not in ("UR10", "UR10E") or not re.search(r"\b5\.\d+", result["version"]):
        raise RuntimeError(f"expected UR10e with PolyScope 5, got {result}")
    if result["remote"].lower() != "true":
        raise RuntimeError("select Remote Control on the UR teach pendant")
    return result


@contextmanager
def _robot_lock(host: str):
    """Shared through Compose's data bind mount, including separate CLI runs."""
    directory = Path("data/.robot-locks")
    directory.mkdir(parents=True, exist_ok=True)
    # Resolve aliases so a hostname and its address cannot take different locks.
    address = socket.gethostbyname(host)
    key = hashlib.sha256(address.encode()).hexdigest()
    with (directory / f"{key}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another ViKi process controls {host}") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _receive(host: str, frequency: int):
    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError("UR replay needs ur-rtde; rebuild the viki Docker image") from exc
    return RTDEReceiveInterface(host, frequency)


def _control(host: str, frequency: int):
    from rtde_control import RTDEControlInterface

    return RTDEControlInterface(host, frequency)


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise RuntimeError(f"invalid RTDE {name}")
    return array


def _ready(receive) -> None:
    if not receive.isConnected():
        raise RuntimeError("RTDE receive disconnected")
    if receive.getSafetyMode() not in (1, 2):
        raise RuntimeError(f"robot safety mode is {receive.getSafetyMode()}, expected normal or reduced")
    if receive.getRobotMode() != 7:
        raise RuntimeError("robot must be powered on with brakes released")


def inspect_ur10e(host: str, *, start_q=None, tolerance: float = 0.01) -> dict:
    """Read-only live connection check: never constructs RTDEControlInterface."""
    if not host:
        raise ValueError("provide --host (robot IP from the teach pendant)")
    identity = _dashboard_info(host)
    receive = _receive(host, 125)
    try:
        _ready(receive)
        q = _vector(receive.getActualQ(), "actual_q")
        qd = _vector(receive.getActualQd(), "actual_qd")
        result = {"identity": identity, "q_rad": q.tolist(), "qd_rad_s": qd.tolist(),
                  "safety_mode": receive.getSafetyMode(), "runtime_state": receive.getRuntimeState()}
        if start_q is not None:
            delta = float(np.max(np.abs(q - np.asarray(start_q))))
            result.update(start_error_rad=delta, at_start=delta <= tolerance)
        return result
    finally:
        receive.disconnect()


class ReplayAborted(RuntimeError):
    def __init__(self, cause: str, log: ProprioLog, *, intentional: bool = False):
        super().__init__(cause)
        self.log = log
        self.intentional = intentional


class UR10eDriver(RobotDriver):
    name = "ur10e"

    def __init__(self, *, host: str, options: URReplayOptions | None = None, stop_event=None):
        self.host = host
        self.options = options or URReplayOptions()
        self.options.validate()
        if not host:
            raise ValueError("UR10e replay requires --host")
        if not self.options.arm_only:
            raise ValueError("no hardware gripper driver is configured; use --arm-only for arm bring-up")
        self.identity = {}
        self.stop_event = stop_event or threading.Event()

    def _check_stop(self):
        if self.stop_event.is_set():
            raise RuntimeError("operator requested replay stop")

    def execute(self, q_traj, gripper, dt: float) -> ProprioLog:
        try:
            return self._execute(q_traj, gripper, dt)
        except ReplayAborted:
            raise
        except (Exception, KeyboardInterrupt) as exc:
            # Connection/lock failures also produce a durable rejected attempt.
            empty = ProprioLog(np.empty((0, 6)), np.empty(0), np.empty(0))
            raise ReplayAborted(str(exc) or "interrupted", empty) from exc

    def _execute(self, q_traj, gripper, dt: float) -> ProprioLog:
        q = np.asarray(q_traj, dtype=np.float64)
        validate_commands(q, np.asarray(gripper), dt)
        times, targets, _, _ = servo_commands(q, dt, self.options)
        rows = []
        with _motion_signals(), _robot_lock(self.host):
            self._check_stop()
            self.identity = _dashboard_info(self.host)
            receive = _receive(self.host, self.options.frequency)
            control = None
            pending = None

            def finish_sample():
                nonlocal pending
                if pending is None:
                    return
                index = pending
                old = rows[index]
                attained = _vector(receive.getActualQ(), "actual_q")
                velocity = _vector(receive.getActualQd(), "actual_qd")
                tcp = _vector(receive.getActualTCPPose(), "actual_tcp_pose")
                timestamp = float(receive.getTimestamp())
                residual = float(np.max(np.abs(attained - old[3])))
                rows[index] = (*old[:2], timestamp, *old[3:5], attained, velocity,
                               tcp, receive.getSafetyMode(), residual, old[10])
                pending = None
            try:
                _ready(receive)
                if receive.getRuntimeState() not in (1, 4):
                    raise RuntimeError("stop the existing robot program before ViKi replay")
                if np.max(np.abs(_vector(receive.getActualQd(), "actual_qd"))) > 0.01:
                    raise RuntimeError("robot must be stationary before replay")
                actual = _vector(receive.getActualQ(), "actual_q")
                delta = float(np.max(np.abs(actual - q[0])))
                if delta > self.options.start_tolerance:
                    raise RuntimeError(
                        f"start pose mismatch: {delta:.4g} rad; position robot at the checked start_q on the pendant"
                    )
                # Connecting this interface uploads a control script. All
                # read-only and start-pose checks therefore precede it.
                self._check_stop()
                control = _control(self.host, self.options.frequency)
                if not control.isConnected():
                    raise RuntimeError("RTDE control disconnected")
                # Check each archived waypoint against installed UR safety
                # limits. These are discrete checks, not environment collision.
                for waypoint in q:
                    self._check_stop()
                    if not control.isJointsWithinSafetyLimits(waypoint.tolist()):
                        raise RuntimeError("plan violates the controller's installed safety limits")
                _ready(receive)
                if np.max(np.abs(_vector(receive.getActualQ(), "actual_q") - q[0])) > self.options.start_tolerance:
                    raise RuntimeError("robot moved during preflight")
                if not control.setWatchdog(10):
                    raise RuntimeError("could not enable controller communication watchdog")
                period = 1 / self.options.frequency
                last_timestamp = float(receive.getTimestamp())
                if not np.isfinite(last_timestamp):
                    raise RuntimeError("invalid RTDE timestamp")
                fresh_at = time.monotonic()
                epoch = fresh_at
                for scheduled, target in zip(times, targets):
                    cycle = control.initPeriod()
                    # Finalize the preceding command at the beginning of this
                    # period. Logging/getters stay inside the period budget,
                    # rather than accumulating drift after every waitPeriod.
                    finish_sample()
                    self._check_stop()
                    host_time = time.monotonic()
                    if host_time - epoch - scheduled > self.options.max_lateness:
                        raise RuntimeError("control deadline missed; aborting instead of catching up")
                    _ready(receive)
                    if not control.isConnected() or not control.isProgramRunning():
                        raise RuntimeError("RTDE control program stopped or disconnected")
                    timestamp = float(receive.getTimestamp())
                    if not np.isfinite(timestamp) or timestamp < last_timestamp:
                        raise RuntimeError("invalid or reversed RTDE controller timestamp")
                    if timestamp > last_timestamp:
                        fresh_at = time.monotonic()
                        last_timestamp = timestamp
                    if time.monotonic() - fresh_at > 0.1:
                        raise RuntimeError("RTDE state is stale")
                    if rows and rows[-1][9] > self.options.tracking_tolerance:
                        raise RuntimeError("joint tracking fault after command")
                    if self.options.stop_after is not None and host_time - epoch >= self.options.stop_after:
                        raise ReplayAborted("test_duration_limit", self._log(rows, q, dt), intentional=True)
                    observed = _vector(receive.getActualQ(), "actual_q")
                    if np.max(np.abs(observed - target)) > self.options.tracking_tolerance:
                        raise RuntimeError("joint tracking fault before command")
                    if not control.kickWatchdog():
                        raise RuntimeError("watchdog kick failed")
                    unknown = np.full(6, np.nan)
                    # Retain attempted commands even when feedback or the
                    # command acknowledgement is lost. Unknown states are NaN.
                    rows.append((scheduled, host_time, np.nan, target.copy(), observed,
                                 unknown.copy(), unknown.copy(), unknown.copy(),
                                 receive.getSafetyMode(), np.nan, False))
                    pending = len(rows) - 1
                    accepted = control.servoJ(target.tolist(), 0.0, 0.0, period, self.options.lookahead_time, self.options.gain)
                    rows[pending] = (*rows[pending][:10], bool(accepted))
                    if not accepted:
                        raise RuntimeError("servoJ command rejected")
                    control.waitPeriod(cycle)
                finish_sample()
                _ready(receive)
                if rows[-1][9] > self.options.tracking_tolerance:
                    raise RuntimeError("joint tracking fault after command")
                return self._log(rows, q, dt)
            except ReplayAborted:
                raise
            except (Exception, KeyboardInterrupt) as exc:
                try:
                    finish_sample()
                except Exception:
                    logger.warning("final RTDE sample unavailable", exc_info=True)
                raise ReplayAborted(str(exc) or "interrupted", self._log(rows, q, dt)) from exc
            finally:
                # Try every cleanup operation even if the first one fails.
                cleanup_errors = []
                if control is not None:
                    for operation in (control.servoStop, control.stopScript, control.disconnect):
                        try:
                            if operation() is False:
                                raise RuntimeError(f"{operation.__name__} returned false")
                        except Exception as exc:
                            cleanup_errors.append(str(exc))
                            logger.exception("UR cleanup failed")
                try:
                    receive.disconnect()
                except Exception:
                    logger.exception("RTDE receive disconnect failed")
                if cleanup_errors:
                    raise ReplayAborted("UR cleanup failed: " + "; ".join(cleanup_errors), self._log(rows, q, dt))

    def _log(self, rows, q: np.ndarray, dt: float) -> ProprioLog:
        names = ("scheduled_time_s", "command_host_monotonic_s", "controller_timestamp_s",
                 "q_commanded", "q_observed", "q_attained", "qd_attained",
                 "tcp_pose_robot", "safety_mode", "controller_residual", "command_accepted")
        telemetry = {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(names)}
        for name in ("q_commanded", "q_observed", "q_attained", "qd_attained", "tcp_pose_robot"):
            telemetry[name] = telemetry[name].reshape(-1, 6)
        if not rows:
            return ProprioLog(np.empty((0, 6)), np.empty(0), np.empty(0), telemetry)
        # Summary only: nearest measurement after each source-frame time.
        # Raw control-rate data above remains the source for policy work.
        source_times = np.arange(len(q)) * dt / self.options.speed_scale
        available = source_times <= telemetry["scheduled_time_s"][-1] + 1e-9
        indices = np.minimum(
            np.searchsorted(telemetry["scheduled_time_s"], source_times[available]), len(rows) - 1,
        )
        return ProprioLog(
            telemetry["q_attained"][indices], np.full(len(indices), np.nan),
            telemetry["controller_residual"][indices], telemetry,
        )
