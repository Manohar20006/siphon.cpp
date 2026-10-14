"""Runner: start/stop/monitor llama-server as a managed process.

Backends (selected automatically, override with SIPHON_DRIVER):
  tmux    - detachable session (survives disconnect), scriptable, attachable
  direct  - plain child process with pid file (fallback if tmux missing)
  systemd - manage existing --user units (e.g. siphon-qwen3.6.service)

Verified launch path (bypasses re-implementation): reuse the project's own
run-*.sh scripts with CONFIG=<materialized env>, exactly like the docs do.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from . import execution
from . import monitor
from .profiles import Profile


@dataclass
class ServerStatus:
    running: bool = False
    driver: str = ""
    name: str = ""
    pid: int | None = None
    health: monitor.HealthState = field(default_factory=monitor.HealthState)
    metrics: monitor.MetricsSnapshot = field(default_factory=monitor.MetricsSnapshot)


_DRIVER: str | None = None
_DIRECT_PROCESSES: dict[int, subprocess.Popen] = {}


def driver_name() -> str:
    """Resolve the active driver once."""
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = os.environ.get("SIPHON_DRIVER", "").strip().lower()
        if not _DRIVER:
            _DRIVER = "tmux" if shutil.which("tmux") else "direct"
    return _DRIVER


def tmux_session_name(profile: str) -> str:
    return f"siphon-{profile}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _sig_to_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _wait_pid_exit(pid: int, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    return False


def _tmux_pane_pid(profile: str) -> int | None:
    res = subprocess.run(
        ["tmux", "list-panes", "-t", tmux_session_name(profile), "-F", "#{pane_pid}"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return None
    for line in res.stdout.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
def status(profile: Profile, *, include_metrics: bool = True) -> ServerStatus:
    st = ServerStatus(name=profile.name, running=False, driver=driver_name())

    if driver_name() == "tmux":
        ok = subprocess.run(
            ["tmux", "has-session", "-t", tmux_session_name(profile.name)],
            capture_output=True,
        ).returncode == 0
        if ok:
            st.running = True
            pid = _tmux_pane_pid(profile.name)
            st.pid = pid or None

    elif driver_name() == "direct":
        pid_path = paths.run_dir() / f"{profile.name}.pid"
        if pid_path.exists():
            pid = int(pid_path.read_text().strip() or "0")
            if _pid_alive(pid):
                st.running = True
                st.pid = pid
            else:
                pid_path.unlink(missing_ok=True)

        # pid file may be stale/missing but the server could still be alive
        llama_pids = _find_llama_pids(profile)
        if llama_pids:
            st.running = True
            st.pid = st.pid or llama_pids[0]

    elif driver_name() == "systemd":
        res = subprocess.run(
            ["systemctl", "--user", "is-active", tmux_session_name(profile.name)],
            capture_output=True, text=True,
        )
        st.running = res.returncode == 0 and res.stdout.strip() == "active"

    if st.running:
        st.health = monitor.fetch_health(profile.base_url)
        if include_metrics:
            st.metrics = monitor.fetch_metrics(profile.base_url)
    return st


def running_profiles(exclude: str | None = None) -> list[Profile]:
    """Return profiles whose server appears to be running (any driver)."""
    from .profiles import discover
    out: list[Profile] = []
    for p in discover().values():
        if exclude and p.name == exclude:
            continue
        if status(p, include_metrics=False).running:   # /metrics queues behind a long prompt; unused here
            out.append(p)
    return out


def up(profile: Profile, *, dry_run: bool = False) -> tuple[bool, str]:
    """Start the server for a profile. Idempotent: no-op if already running."""
    try:
        execution_mode = execution.mode(profile)
    except ValueError as exc:
        return False, f"profile invalid:\n  {exc}"

    if execution_mode == "cpu" and driver_name() == "systemd":
        return False, "CPU launches are not supported by the systemd driver; use direct or tmux."

    if not dry_run:
        st = status(profile)
        if st.running:
            return True, (
                f"{profile.name} already running "
                f"(health={st.health.status})"
            )

        others = running_profiles(profile.name)
        if others:
            names = ", ".join(p.name for p in others)
            return False, (
                f"another server is using shared compute resources: {names}. Stop it first "
                "(only one model can run at a time)."
            )

    issues = profile.validate()
    if issues:
        return False, "profile invalid:\n  " + "\n  ".join(issues)

    if execution_mode == "cpu":
        try:
            cmd, env = execution.cpu_launch(profile)
        except (OSError, ValueError) as exc:
            return False, f"CPU launch failed: {exc}"
        env["SIPHON_PROFILE"] = profile.name
        if dry_run:
            import shlex
            projected = execution.preview_environment(profile)
            env_args = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in projected.items()
            )
            return True, "DRY RUN:\n  env " + env_args + " " + shlex.join(cmd)
    else:
        env_file = profile.materialize_env()
        script = paths.repo_root() / profile.script_rel
        if not script.exists():
            return False, f"launcher script not found: {script}"

        cmd = ["bash", str(script)]
        env = dict(os.environ)
        env["CONFIG"] = str(env_file)

        if dry_run:
            import shlex
            return True, "DRY RUN:\n  CONFIG=" + str(env_file) + " " + " ".join(
                shlex.quote(str(c)) for c in cmd
            )

    if driver_name() == "tmux":
        return _tmux_up(profile, cmd, env)
    elif driver_name() == "systemd":
        return _systemd_up(profile)
    return _direct_up(profile, cmd, env)
def _tmux_up(profile: Profile, cmd: list[str], env: dict[str, str]) -> tuple[bool, str]:
    import shlex
    import subprocess as sp
    sess = tmux_session_name(profile.name)
    if sp.run(["tmux", "has-session", "-t", sess], capture_output=True).returncode == 0:
        return True, f"tmux session {sess} already running"

    env["SIPHON_PROFILE"] = profile.name
    log_file = profile.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    inner = " ".join(shlex.quote(str(c)) for c in cmd)
    pipe = f"2>&1 | tee -a {shlex.quote(str(log_file))}"

    script_text = f"#!/usr/bin/env bash\n{envs} {inner} {pipe}\n"
    cmd_path = paths.cmd_dir() / f"{profile.name}.sh"
    cmd_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_path.write_text(script_text, encoding="utf-8")
    cmd_path.chmod(0o755)

    res = sp.run(
        ["tmux", "new-session", "-d", "-s", sess,
         "exec " + shlex.join(["env", f"SIPHON_PROFILE={profile.name}", str(cmd_path)])],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return False, f"tmux failed: {res.stdout} {res.stderr}"
    return True, f"started {profile.name} in tmux session {sess}"


def _direct_up(profile: Profile, cmd: list[str], env: dict[str, str]) -> tuple[bool, str]:
    for old_pid, old_proc in list(_DIRECT_PROCESSES.items()):
        if old_proc.poll() is not None:
            _DIRECT_PROCESSES.pop(old_pid, None)
    log_file = profile.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "ab")
    launch_env = dict(env)
    launch_env["SIPHON_PROFILE"] = profile.name
    # nohup + own session + detached stdin: the server survives the parent
    # terminal/CLI process exiting, like `nohup ... & disown`
    wrapped = ["nohup", *cmd]
    try:
        proc = subprocess.Popen(
            wrapped, env=launch_env, stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        fh.close()
        return False, f"failed to start: {exc}"
    finally:
        if not fh.closed:
            fh.close()

    run_dir = paths.run_dir()
    pid_path = run_dir / f"{profile.name}.pid"
    owner_path = run_dir / f"{profile.name}.owner.json"
    temporary = owner_path.with_name(f"{owner_path.name}.{proc.pid}.tmp")
    try:
        stat = _proc_stat(proc.pid)
        if stat is None:
            _sig_to_pid(proc.pid, signal.SIGKILL)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            return False, f"process exited during launch (pid {proc.pid})"
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        owner = {
            "pid": proc.pid,
            "pgid": stat[2],
            "starttime": stat[4],
            "profile": profile.name,
            "config": launch_env.get("CONFIG", ""),
        }
        temporary.write_text(json.dumps(owner), encoding="utf-8")
        os.replace(temporary, owner_path)
    except OSError as exc:
        _sig_to_pid(proc.pid, signal.SIGKILL)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        temporary.unlink(missing_ok=True)
        pid_path.unlink(missing_ok=True)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        return False, f"failed to record process ownership: {exc}"
    _DIRECT_PROCESSES[proc.pid] = proc
    return True, f"started {profile.name} (pid {proc.pid}, direct driver)"


def _systemd_up(profile: Profile) -> tuple[bool, str]:
    res = subprocess.run(
        ["systemctl", "--user", "start", tmux_session_name(profile.name)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return False, f"systemd start failed: {res.stderr}"
    return True, f"started via systemd unit {tmux_session_name(profile.name)}"
def _find_llama_pids(profile: Profile) -> list[int]:
    """Find live llama-server processes for this profile (by model path match).

    The direct driver records the bash launcher's pid, but llama-server is a
    child of it — when the wrapper is nohup'd/detached, SIGINT to the wrapper
    may not reach the server. Match the MODEL path in cmdline.

    NOTE: models/ entries may be symlinks — the run script passes the
    repo-relative expansion (symlink NOT followed), so match BOTH the raw
    expanded path and the fully-resolved (symlink-followed) path.
    """
    import re as _re
    root = paths.repo_root()

    candidates: set[str] = set()
    raw = profile.env.get("MODEL", "")
    if raw:
        cand = raw if raw.startswith("/") else str(root / raw)
        candidates.add(cand)
        candidates.add(str(Path(cand).resolve()))
        candidates.add(str(Path(raw).resolve()))

    pids: set[int] = set()
    for cand in candidates:
        if not cand:
            continue
        pat = _re.escape(cand)
        try:
            res = subprocess.run(["pgrep", "-af", pat],
                                 capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        for line in res.stdout.splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2 and "llama-server" in parts[1]:
                try:
                    pid = int(parts[0])
                except ValueError:
                    continue
                # several profiles may share a model; a server launched for another profile is not ours
                owner = _process_profile(pid)
                if owner is None or owner == profile.name:
                    pids.add(pid)
    return sorted(pids)


def _process_profile(pid: int) -> str | None:
    """SIPHON_PROFILE of a process launched by Studio (None if unknown or not launched by Studio)."""
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    for item in environ:
        if item.startswith(b"SIPHON_PROFILE="):
            return item[len(b"SIPHON_PROFILE="):].decode("utf-8", "replace")
    return None


_SHUTDOWN_PHASE_SECONDS = 0.75
_SHUTDOWN_POLL_SECONDS = 0.05


def _proc_stat(pid: int) -> tuple[str, int, int, int, int] | None:
    """Read state, parent, process group, session, and start time from procfs."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2:].split()
        return fields[0], int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19])
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return None


def _proc_group_members(pgid: int) -> list[tuple[int, tuple[str, int, int, int, int]]]:
    try:
        entries = Path("/proc").iterdir()
        members = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            stat = _proc_stat(int(entry.name))
            if stat is not None and stat[2] == pgid:
                members.append((int(entry.name), stat))
        return members
    except OSError as exc:
        raise RuntimeError(f"cannot inspect process group {pgid}: {exc}") from exc


def _process_has_owner(pid: int, profile: str, config: str) -> bool:
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, OSError):
        return False

    profile_value = f"SIPHON_PROFILE={profile}".encode()
    if profile_value in environ or profile_value in cmdline:
        return True
    if not config:
        return False
    config_value = f"CONFIG={config}".encode()
    return config_value in environ or config_value in cmdline or (
        b"CONFIG" in cmdline and cmdline[cmdline.index(b"CONFIG") + 1:]
        and cmdline[cmdline.index(b"CONFIG") + 1] == config.encode()
    )


def _active_group_members(pgid: int) -> list[tuple[int, tuple[str, int, int, int, int]]]:
    return [(pid, stat) for pid, stat in _proc_group_members(pgid) if stat[0] != "Z"]


def _terminate_owned_group(
    pgid: int, profile: str, config: str,
    verified: dict[int, int] | None = None,
) -> str | None:
    """Escalate within a verified, dedicated process group and confirm exit."""
    if pgid <= 1 or pgid == os.getpgrp():
        return f"refusing to signal non-dedicated process group {pgid}"
    verified = dict(verified or {})

    def unowned_members(members):
        unowned = []
        for pid, stat in members:
            if verified.get(pid) == stat[4]:
                continue
            if _process_has_owner(pid, profile, config):
                verified[pid] = stat[4]
            else:
                unowned.append(pid)
        return unowned

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        members = _active_group_members(pgid)
        if not members:
            return None
        unowned = unowned_members(members)
        if unowned:
            return f"process group {pgid} contains unowned processes: {unowned}"
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return None
        except OSError as exc:
            return f"cannot signal owned process group {pgid}: {exc}"

        deadline = time.monotonic() + _SHUTDOWN_PHASE_SECONDS
        while time.monotonic() < deadline:
            members = _active_group_members(pgid)
            if not members:
                return None
            unowned = unowned_members(members)
            if unowned:
                return f"process group {pgid} contains unowned processes: {unowned}"
            time.sleep(_SHUTDOWN_POLL_SECONDS)
    if _active_group_members(pgid):
        return f"owned process group {pgid} remained active after SIGKILL"
    return None


def _direct_state_paths(profile: Profile) -> tuple[Path, Path]:
    run_dir = paths.run_dir()
    return run_dir / f"{profile.name}.pid", run_dir / f"{profile.name}.owner.json"


def _remove_direct_state(pid_path: Path, owner_path: Path) -> str | None:
    try:
        pid_path.unlink(missing_ok=True)
        owner_path.unlink(missing_ok=True)
    except OSError as exc:
        return f"cannot remove direct ownership state: {exc}"
    return None


def _shutdown_direct(profile: Profile) -> str | None:
    pid_path, owner_path = _direct_state_paths(profile)
    if not pid_path.exists() and not owner_path.exists():
        return None

    config = str(paths.env_dir() / f"{profile.name}.env")
    if owner_path.exists():
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            pid = int(owner["pid"])
            pgid = int(owner["pgid"])
            starttime = int(owner["starttime"])
            owner_profile = owner["profile"]
            config = owner.get("config", "") or config
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return f"invalid direct ownership metadata: {exc}"
        if owner_profile != profile.name or pid <= 1 or pgid <= 1 or starttime <= 0:
            return "direct ownership metadata does not match this profile"
        if pid_path.exists():
            try:
                if int(pid_path.read_text(encoding="utf-8").strip()) != pid:
                    return "direct PID file does not match ownership metadata"
            except (OSError, ValueError) as exc:
                return f"invalid direct PID file: {exc}"
        stat = _proc_stat(pid)
        if stat is not None:
            if stat[4] != starttime:
                return f"direct PID {pid} identity changed; refusing to signal it"
            if stat[2] != pgid:
                return f"owned PID {pid} changed process group; refusing to signal it"
        members = _active_group_members(pgid)
        if members:
            unowned = [member_pid for member_pid, _member_stat in members
                       if not _process_has_owner(member_pid, profile.name, config)]
            if unowned:
                return f"process group {pgid} contains unowned processes: {unowned}"
            error = _terminate_owned_group(pgid, profile.name, config)
            if error:
                return error
    else:
        if not pid_path.exists():
            return "direct PID file is missing ownership metadata"
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            return f"invalid legacy PID file: {exc}"
        if pid <= 1:
            return "invalid legacy PID value"
        stat = _proc_stat(pid)
        if stat is not None:
            if not _process_has_owner(pid, profile.name, config):
                return f"legacy PID {pid} is live but has no verified profile ownership"
            pgid = stat[2]
            error = _terminate_owned_group(pgid, profile.name, config)
            if error:
                return error
        else:
            # Direct launches create a new session with PGID == launcher PID.
            # The group can still contain a model after that launcher exits.
            error = _terminate_owned_group(pid, profile.name, config)
            if error:
                return error

    error = _remove_direct_state(pid_path, owner_path)
    if not error:
        owned_proc = _DIRECT_PROCESSES.pop(pid, None)
        if owned_proc is not None:
            try:
                owned_proc.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
    return error


def _tmux_has_session(profile: Profile) -> tuple[bool, str | None]:
    session = tmux_session_name(profile.name)
    result = subprocess.run(
        ["tmux", "has-session", "-t", session], capture_output=True, text=True, timeout=5,
    )
    if result.returncode == 0:
        return True, None
    detail = (result.stderr or result.stdout).strip()
    absent_messages = ("can't find session", "no server running", "no current client")
    if any(message in detail.lower() for message in absent_messages):
        return False, None
    if "error connecting to" in detail.lower() and "no such file or directory" in detail.lower():
        return False, None
    return False, f"tmux has-session failed: {detail or result.returncode}"


def _shutdown_tmux(profile: Profile) -> str | None:
    session = tmux_session_name(profile.name)
    exists, error = _tmux_has_session(profile)
    if error or not exists:
        return error
    panes = subprocess.run(
        ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
        capture_output=True, text=True, timeout=5,
    )
    if panes.returncode != 0:
        return f"tmux pane lookup failed: {(panes.stderr or panes.stdout).strip()}"
    pane_pids: set[int] = set()
    for line in panes.stdout.splitlines():
        try:
            pane_pids.add(int(line.strip()))
        except ValueError:
            continue
    if not pane_pids:
        return f"tmux session {session} has no verifiable pane process"

    config = str(paths.env_dir() / f"{profile.name}.env")
    verified: dict[int, int] = {}
    groups: set[int] = set()
    for pid in pane_pids:
        stat = _proc_stat(pid)
        if stat is None or stat[0] == "Z":
            return f"tmux pane PID {pid} is not available for ownership verification"
        if not _process_has_owner(pid, profile.name, config):
            # Older launch scripts tagged only the server, not bash/tee.
            # Accept the exact managed wrapper, never a model-path match.
            try:
                args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except OSError:
                args = []
            script = str(paths.cmd_dir() / f"{profile.name}.sh").encode()
            if script not in args:
                return f"tmux pane PID {pid} has no verified profile ownership"
        verified[pid] = stat[4]
        groups.add(stat[2])
    # Snapshot descendants before signalling parents. Retain PID/start-time
    # identities when children become orphans, including pipeline loggers.
    processes = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            stat = _proc_stat(int(entry.name))
            if stat is not None and stat[0] != "Z":
                processes[int(entry.name)] = stat
    changed = True
    while changed:
        changed = False
        for pid, stat in processes.items():
            parent = stat[1]
            if pid not in verified and parent in verified:
                parent_stat = processes.get(parent)
                if parent_stat is not None and parent_stat[4] == verified[parent]:
                    verified[pid] = stat[4]
                    groups.add(stat[2])
                    changed = True
    for pgid in groups:
        error = _terminate_owned_group(pgid, profile.name, config, verified)
        if error:
            return error

    exists, error = _tmux_has_session(profile)
    if error:
        return error
    if exists:
        killed = subprocess.run(
            ["tmux", "kill-session", "-t", session], capture_output=True, text=True, timeout=5,
        )
        if killed.returncode != 0:
            exists, probe_error = _tmux_has_session(profile)
            if probe_error:
                return probe_error
            if exists:
                return f"tmux kill-session failed: {(killed.stderr or killed.stdout).strip()}"
    exists, error = _tmux_has_session(profile)
    if error:
        return error
    if exists:
        return f"tmux session {session} remained after shutdown"
    return None


def _shutdown_systemd(profile: Profile) -> str | None:
    unit = tmux_session_name(profile.name)

    def inspect_unit():
        result = subprocess.run(
            ["systemctl", "--user", "show", unit,
             "--property=LoadState,ActiveState,MainPID,ControlGroup"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(f"systemd unit lookup failed: {(result.stderr or result.stdout).strip()}")
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if not {"LoadState", "ActiveState", "MainPID"}.issubset(values):
            raise RuntimeError("systemd unit lookup returned incomplete state")
        return values

    before = inspect_unit()
    if before["LoadState"] == "not-found":
        return None
    if before["ActiveState"] in ("inactive", "failed") and before["MainPID"] == "0":
        if not _systemd_group_active(before.get("ControlGroup", "")):
            return None
    result = subprocess.run(
        ["systemctl", "--user", "stop", unit], capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return f"systemd stop failed for {unit}: {(result.stderr or result.stdout).strip()}"
    after = inspect_unit()
    if (after["ActiveState"] not in ("inactive", "failed") or after["MainPID"] != "0"
            or _systemd_group_active(before.get("ControlGroup", ""))
            or _systemd_group_active(after.get("ControlGroup", ""))):
        return f"systemd unit {unit} remained active after stop"
    return None


def _systemd_group_active(control_group: str) -> bool:
    """Verify remaining children as well as MainPID for a user service."""
    if not control_group:
        return False
    root = Path("/sys/fs/cgroup")
    group = (root / control_group.lstrip("/")).resolve()
    if group == root or not group.is_relative_to(root):
        raise RuntimeError("invalid systemd control group")
    if not group.exists():
        return False
    for procs in (group / "cgroup.procs", *group.rglob("cgroup.procs")):
        try:
            pids = procs.read_text().splitlines()
        except FileNotFoundError:
            continue
        for value in pids:
            stat = _proc_stat(int(value))
            if stat is not None and stat[0] != "Z":
                return True
    return False


def shutdown(profiles: Iterable[Profile]) -> list[str]:
    """Stop all supplied desktop-managed launches and report every failure."""
    errors: list[str] = []
    seen: set[str] = set()
    for profile in profiles:
        if profile.name in seen:
            continue
        seen.add(profile.name)
        driver = driver_name()
        # Persisted direct ownership remains valid after a driver change.
        handlers = [_shutdown_direct]
        if driver == "tmux":
            handlers.append(_shutdown_tmux)
        elif driver == "systemd":
            handlers.append(_shutdown_systemd)
        elif driver != "direct":
            errors.append(f"{profile.name}: unsupported runner driver {driver!r}")
        for handler in handlers:
            try:
                error = handler(profile)
            except Exception as exc:
                error = f"shutdown failed: {exc}"
            if error:
                errors.append(f"{profile.name}: {error}")
    return errors


def down(profile: Profile, *, force: bool = False) -> tuple[bool, str]:
    st = status(profile)
    llama_pids = _find_llama_pids(profile)
    if not st.running and not llama_pids:
        return True, f"{profile.name} already stopped"
    if st.running and not (force or st.health.ok):
        return False, (
            f"{profile.name} is not healthy ({st.health.status}); "
            "use --force to stop anyway"
        )

    if driver_name() == "tmux":
        if st.pid is not None:
            _sig_to_pid(st.pid, signal.SIGINT)   # graceful
            for pid in _find_llama_pids(profile):
                _sig_to_pid(pid, signal.SIGINT)
            _wait_pid_exit(st.pid, timeout=15)
        subprocess.run(["tmux", "kill-session", "-t", tmux_session_name(profile.name)],
                       capture_output=True)
        return True, f"stopped {profile.name} (tmux session killed)"

    if driver_name() == "systemd":
        subprocess.run(["systemctl", "--user", "stop", tmux_session_name(profile.name)],
                       capture_output=True)
        return True, f"stopped systemd unit {tmux_session_name(profile.name)}"

    # direct driver
    if st.pid:
        _sig_to_pid(st.pid, signal.SIGINT)
    for pid in _find_llama_pids(profile):
        _sig_to_pid(pid, signal.SIGINT)
    if st.pid:
        _wait_pid_exit(st.pid, timeout=20)
    # clear the pid file
    pid_path = paths.run_dir() / f"{profile.name}.pid"
    pid_path.unlink(missing_ok=True)
    return True, f"stopped {profile.name}"


def restart(profile: Profile) -> tuple[bool, str]:
    ok, msg = down(profile, force=True)
    time.sleep(1.0)
    return up(profile)


def wait_healthy(
    profile: Profile,
    timeout: float = 600.0,
    interval: float = 1.0,
    cancel_event: threading.Event | None = None,
) -> bool:
    """Block until /health reports ok (return True) or timeout (False)."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        if cancel_event is not None and cancel_event.is_set():
            return False
        state = status(profile)
        if cancel_event is not None and cancel_event.is_set():
            return False
        if state.health.ok:
            return True
        if not state.running:
            if last != "down":
                print(f"  waiting for {profile.name} ... [process exited]")
            return False
        if state.health.status != last:
            print(f"  waiting for {profile.name} ... [{state.health.status}]")
            last = state.health.status
        if cancel_event is None:
            time.sleep(interval)
        elif cancel_event.wait(interval):
            return False
    return False
    return None
