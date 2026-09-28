"""Runner: start/stop/monitor llama-server as a managed process.

Backends (selected automatically, override with SIPHON_DRIVER):
  tmux    - detachable session (survives disconnect), scriptable, attachable
  direct  - plain child process with pid file (fallback if tmux missing)
  systemd - manage existing --user units (e.g. siphon-qwen3.6.service)

Verified launch path (bypasses re-implementation): reuse the project's own
run-*.sh scripts with CONFIG=<materialized env>, exactly like the docs do.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
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
        if status(p).running:
            out.append(p)
    return out


def up(profile: Profile, *, dry_run: bool = False) -> tuple[bool, str]:
    """Start the server for a profile. Idempotent: no-op if already running."""
    if not dry_run:
        st = status(profile)
        if st.running:
            return True, (
                f"{profile.name} already running "
                f"(health={st.health.status})"
            )

        # GPU contention guard: only one model can run on the small GPU at a time.
        others = running_profiles(profile.name)
        if others:
            names = ", ".join(p.name for p in others)
            return False, (
                f"another server is using the GPU: {names}. Stop it first "
                "(only one model can run on this 6 GB GPU)."
            )

    issues = profile.validate()
    if issues:
        return False, "profile invalid:\n  " + "\n  ".join(issues)

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
        ["tmux", "new-session", "-d", "-s", sess, shlex.quote(str(cmd_path))],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return False, f"tmux failed: {res.stdout} {res.stderr}"
    return True, f"started {profile.name} in tmux session {sess}"


def _direct_up(profile: Profile, cmd: list[str], env: dict[str, str]) -> tuple[bool, str]:
    log_file = profile.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "ab")
    # nohup + own session + detached stdin: the server survives the parent
    # terminal/CLI process exiting, like `nohup ... & disown`
    wrapped = ["nohup", *cmd]
    try:
        proc = subprocess.Popen(
            wrapped, env=env, stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        fh.close()
        return False, f"failed to start: {exc}"

    pid_path = paths.run_dir() / f"{profile.name}.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
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
                    pids.add(int(parts[0]))
                except ValueError:
                    continue
    return sorted(pids)


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


def wait_healthy(profile: Profile, timeout: float = 600.0, interval: float = 1.0) -> bool:
    """Block until /health reports ok (return True) or timeout (False)."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        state = status(profile)
        if state.health.ok:
            return True
        if not state.running:
            if last != "down":
                print(f"  waiting for {profile.name} ... [process exited]")
            return False
        if state.health.status != last:
            print(f"  waiting for {profile.name} ... [{state.health.status}]")
            last = state.health.status
        time.sleep(interval)
    return False
    return None
