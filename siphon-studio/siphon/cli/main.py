"""siphon.cli - headless command surface for AI agents and scripts.

Commands (all support --json for machine-readable output):
  list             profiles + models + suites
  up <profile>     start (idempotent; --dry-run preview)
  down <profile>   stop (--all)
  status           running servers (--json)
  logs <profile>   tail the server log (--tail N, --json)
  metrics <profile>via /metrics (--json)
  wait <profile>   block until healthy
  doctor           environment + GDS health checks
  profile          new | clone | rm
  desktop          launch the native Siphon Studio desktop app
  api              local API for your apps: serve | info | keys show|regenerate | apps | connect | disconnect
"""

from __future__ import annotations

import argparse
import json
import sys

from .. import __version__
from ..core.profiles import discover
from ..core import doctor, runner
from ..core import monitor  # noqa: F401  (re-exported for CLI use)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="siphon", description="siphon-studio headless CLI")
    p.add_argument("--version", action="version", version=f"siphon-studio {__version__}")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    sub = p.add_subparsers(dest="cmd")   # optional: no cmd -> open the TUI

    # shared flag, usable both before and after the subcommand (SUPPRESS so
    # the top-level default survives when the flag is omitted here).
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    sub.add_parser("list", parents=[parent], help="list profiles + models + suites")

    up_p = sub.add_parser("up", parents=[parent], help="start a profile")
    up_p.add_argument("profile")
    up_p.add_argument("--dry-run", action="store_true")
    up_p.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                     help="wait up to N sec for /health after starting")

    dn_p = sub.add_parser("down", parents=[parent], help="stop a profile")
    dn_p.add_argument("profile", nargs="?", default=None)
    dn_p.add_argument("--all", action="store_true")
    dn_p.add_argument("--force", action="store_true")

    sub.add_parser("status", parents=[parent], help="show running servers")

    lg = sub.add_parser("logs", parents=[parent], help="tail a profile's log")
    lg.add_argument("profile")
    lg.add_argument("--tail", type=int, default=50)

    mt = sub.add_parser("metrics", parents=[parent], help="show /metrics for a profile")
    mt.add_argument("profile")

    wt = sub.add_parser("wait", parents=[parent], help="block until a profile is healthy")
    wt.add_argument("profile")
    wt.add_argument("--timeout", type=int, default=600)

    sub.add_parser("doctor", parents=[parent], help="run environment health checks")
    sub.add_parser("desktop", parents=[parent], help="launch the native desktop app")

    pf = sub.add_parser("profile", parents=[parent], help="manage app profiles")
    pf_sub = pf.add_subparsers(dest="profile_cmd", required=True)
    new_p = pf_sub.add_parser("new", help="create a new app profile")
    new_p.add_argument("name")
    new_p.add_argument("--from", dest="from_name", default=None,
                       help="seed from an existing profile")
    new_p.add_argument("--script", default="")
    rm_p = pf_sub.add_parser("rm", help="delete an app profile")
    rm_p.add_argument("name")

    ap = sub.add_parser("api", parents=[parent], help="local API gateway + scoped API keys")
    ap_sub = ap.add_subparsers(dest="api_cmd", required=True)
    sv = ap_sub.add_parser("serve", parents=[parent], help="run the API gateway on 127.0.0.1 (foreground)")
    sv.add_argument("--port", type=int, default=0, help="port (default $SIPHON_API_PORT or 8088)")
    ap_sub.add_parser("info", parents=[parent], help="endpoint, running engine and usage snippets")
    ks = ap_sub.add_parser("keys", parents=[parent], help="the decode and decision API keys")
    ks_sub = ks.add_subparsers(dest="keys_cmd", required=True)
    kshow = ks_sub.add_parser("show", parents=[parent], help="print the API keys (created on first use)")
    kshow.add_argument("--scope", choices=["decode", "decision"], default=None)
    kr = ks_sub.add_parser("regenerate", parents=[parent], help="replace a key; the old one stops working")
    kr.add_argument("--scope", required=True, choices=["decode", "decision"])
    ap_sub.add_parser("apps", parents=[parent], help="coding apps that can be connected to the decode API")
    for name, text in (("connect", "point an app at the local decode API"), ("disconnect", "undo connect")):
        cp = ap_sub.add_parser(name, parents=[parent], help=text)
        cp.add_argument("app", choices=["opencode", "claude-code", "codex"])
        if name == "connect":
            cp.add_argument("--profile", default=None, help="decode profile whose model alias to use")

    return p.parse_args(argv)


def _emit(obj, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(obj, indent=2) + "\n")
    else:
        if isinstance(obj, str):
            sys.stdout.write(obj + "\n")
        elif isinstance(obj, list):
            for line in obj:
                sys.stdout.write(str(line) + "\n")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                sys.stdout.write(f"{k}: {v}\n")


def _profile_or_die(profiles: dict, name: str):
    prof = profiles.get(name)
    if prof is None:
        sys.stderr.write(f"unknown profile: {name} (see 'siphon list')\n")
        sys.exit(2)
    return prof
def _cmd_list(args, profs, as_json) -> None:
    from ..core.profiles import find_models
    from ..core.suites import list_suites
    obj = {
        "profiles": [p.to_dict() for p in profs.values()],
        "models": find_models(),
        "suites": list_suites(),
    }
    _emit(obj, as_json)


def _cmd_up(args, profs, as_json) -> int:
    prof = _profile_or_die(profs, args.profile)
    if as_json and not args.dry_run:
        print("starting ...", file=sys.stderr)
    ok, msg = runner.up(prof, dry_run=args.dry_run)
    if args.wait and ok and not args.dry_run:
        ok_wait = runner.wait_healthy(prof, timeout=args.wait)
        msg += f"\nhealthy after wait: {ok_wait}"
        ok = ok and ok_wait
    _emit({"ok": ok, "msg": msg}, as_json)
    return 0 if ok else 1


def _cmd_down(args, profs, as_json) -> int:
    if args.all:
        results: dict[str, dict] = {}
        code = 0
        for name, prof in profs.items():
            ok, msg = runner.down(prof, force=args.force)
            results[name] = {"ok": ok, "msg": msg}
            if not ok:
                code = 1
        _emit(results, as_json)
        return code
    if not args.profile:
        print("down requires a profile or --all")
        return 2
    prof = _profile_or_die(profs, args.profile)
    ok, msg = runner.down(prof, force=args.force)
    _emit({"ok": ok, "msg": msg}, as_json)
    return 0 if ok else 1


def _cmd_status(args, profs, as_json) -> int:
    lines = []
    for name, prof in profs.items():
        st = runner.status(prof)
        lines.append({
            "name": name,
            "running": st.running,
            "health": st.health.status,
            "decode_tps": st.metrics.decode_tps,
            "prefill_tps": st.metrics.prefill_tps,
            "pid": st.pid,
        })
    _emit(lines, as_json)
    return 0


def _cmd_logs(args, profs, as_json) -> int:
    prof = _profile_or_die(profs, args.profile)
    log = prof.log_file
    if not log.exists():
        _emit({"ok": False, "msg": f"no log yet: {log}"}, as_json)
        return 1
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-args.tail:] if args.tail else lines
    if as_json:
        _emit({"ok": True, "lines": tail}, as_json)
    else:
        sys.stdout.write("\n".join(tail) + ("\n" if tail else ""))
    return 0


def _cmd_metrics(args, profs, as_json) -> int:
    prof = _profile_or_die(profs, args.profile)
    st = runner.status(prof)
    if not st.running:
        _emit({"ok": False, "msg": f"{args.profile} not running"}, as_json)
        return 1
    m = st.metrics
    _emit({
        "ok": True,
        "decode_tps": m.decode_tps,
        "prefill_tps": m.prefill_tps,
        "n_decode": m.n_decode,
        "n_predicted": m.n_predicted,
        "n_prompt": m.n_prompt,
        "requests_processing": m.n_processing,
        "health": st.health.status,
        "raw": m.raw,
    }, as_json)
    return 0


def _cmd_wait(args, profs, as_json) -> int:
    prof = _profile_or_die(profs, args.profile)
    ok = runner.wait_healthy(prof, timeout=args.timeout)
    _emit({"ok": ok, "profile": args.profile}, as_json)
    return 0 if ok else 1


def _cmd_doctor(args, as_json) -> int:
    checks = doctor.run_checks()
    if as_json:
        _emit([{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks], as_json)
    else:
        for c in checks:
            print(f"[{'OK' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    return 0 if all(c.ok for c in checks) else 1


def _cmd_profile(args, profs, as_json) -> int:
    from ..core.profiles import Profile, delete_app_profile
    if args.profile_cmd == "new":
        if args.from_name:
            src = profs.get(args.from_name)
            if src is None:
                print(f"unknown source profile: {args.from_name}")
                return 2
            prof = Profile.create_app(args.name, dict(src.env),
                                      script=args.script or src.script_rel)
        else:
            prof = Profile.create_app(args.name, {}, script=args.script)
        prof.save()
        _emit({"ok": True, "msg": f"created app profile {args.name}"}, as_json)
        return 0
    if args.profile_cmd == "rm":
        if delete_app_profile(args.name):
            _emit({"ok": True, "msg": f"deleted {args.name}"}, as_json)
        else:
            _emit({"ok": False, "msg": f"no app profile {args.name}"}, as_json)
            return 1
    return 0


def _cmd_api(args, as_json) -> int:
    from ..api import connect, gateway
    from ..api.keys import KeyStore
    if args.api_cmd == "serve":
        port = args.port or gateway.default_port()
        try:
            gw = gateway.serve(port)
        except OSError as exc:
            sys.stderr.write(f"cannot bind {gateway.HOST}:{port}: {exc}\n")
            return 1
        print(f"Siphon API on http://{gateway.HOST}:{port}  (Ctrl-C to stop)", file=sys.stderr)
        try:
            gw.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            gw.keys.flush()
            gw.server_close()
        return 0
    if args.api_cmd == "info":
        _emit(api_info(), as_json)
        return 0
    store = KeyStore()
    keys_cmd = getattr(args, "keys_cmd", None)
    if keys_cmd == "show":
        scopes = [args.scope] if args.scope else ["decode", "decision"]
        _emit({scope: store.get(scope).key for scope in scopes}, as_json)
        return 0
    if keys_cmd == "regenerate":
        record = store.regenerate(args.scope)
        msg = f"new {record.scope} key (the old one no longer works)"
        if record.scope == "decode":
            updated = connect.refresh(record.key)
            if updated:
                msg += f"; updated connected apps: {', '.join(updated)}"
        _emit({"ok": True, "msg": msg, record.scope: record.key}, as_json)
        return 0
    if args.api_cmd == "apps":
        _emit({app["id"]: {k: app[k] for k in ("installed", "connected", "where", "usage")}
               for app in connect.status()}, as_json)
        return 0
    if args.api_cmd in {"connect", "disconnect"}:
        try:
            if args.api_cmd == "disconnect":
                connect.disconnect(args.app)
                _emit({"ok": True, "msg": f"disconnected {args.app}"}, as_json)
                return 0
            profile = discover().get(args.profile) if args.profile else None
            if args.profile and profile is None:
                raise connect.ConnectError(f"no profile named {args.profile!r}")
            backend = gateway.BackendResolver().current()
            model = (profile.env.get("ALIAS", profile.name) if profile else backend.alias if backend else "siphon")
            context = int((profile.env if profile else {}).get("CTX", 65536))
            target = connect.Target(endpoint=f"http://{gateway.HOST}:{gateway.default_port()}",
                                    key=store.get("decode").key, model=model, context=context)
            _emit({"ok": True, "msg": connect.connect(args.app, target)}, as_json)
            return 0
        except connect.ConnectError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
    return 2


def api_info() -> dict:
    from ..api import gateway
    base = f"http://{gateway.HOST}:{gateway.default_port()}"
    backend = gateway.BackendResolver().current()
    return {
        "endpoint": base,
        "running": None if backend is None else {"profile": backend.profile, "mode": backend.mode,
                                                 "model": backend.alias},
        "decode": {"key_scope": "decode", "openai_base_url": f"{base}/v1",
                   "anthropic_base_url": base,
                   "routes": ["POST /v1/chat/completions", "POST /v1/completions", "POST /v1/responses",
                              "POST /v1/messages", "POST /v1/messages/count_tokens", "GET /v1/models"]},
        "decision": {"key_scope": "decision", "routes": ["POST /v1/decide", "POST /v1/decide/warmup"],
                     "body": {"input": "user text", "question": "what to decide",
                              "options": ["name", {"name": "...", "description": "..."}],
                              "context": "optional, shared across calls", "threshold": 0.9}},
    }


def main() -> int:
    args = _parse_args(sys.argv[1:])

    # no subcommand -> launch the TUI (Graphical Terminal dashboard)
    if args.cmd is None:
        try:
            from ..tui.app import run as run_tui
        except ImportError as exc:
            sys.stderr.write(
                f"TUI unavailable ({exc}). Install it: "
                "python3 -m pip install --user --break-system-packages textual\n"
            )
            return 3
        run_tui()
        return 0

    profs = discover()
    as_json = args.json

    if args.cmd == "list":
        _cmd_list(args, profs, as_json)
    elif args.cmd == "up":
        return _cmd_up(args, profs, as_json)
    elif args.cmd == "down":
        return _cmd_down(args, profs, as_json)
    elif args.cmd == "status":
        return _cmd_status(args, profs, as_json)
    elif args.cmd == "logs":
        return _cmd_logs(args, profs, as_json)
    elif args.cmd == "metrics":
        return _cmd_metrics(args, profs, as_json)
    elif args.cmd == "wait":
        return _cmd_wait(args, profs, as_json)
    elif args.cmd == "doctor":
        return _cmd_doctor(args, as_json)
    elif args.cmd == "desktop":
        try:
            from ..desktop.app import run as run_desktop
        except ImportError as exc:
            sys.stderr.write(
                f"Desktop unavailable ({exc}). GTK4 and WebKitGTK 6.0 are required.\n"
            )
            return 3
        return run_desktop()
    elif args.cmd == "profile":
        return _cmd_profile(args, profs, as_json)
    elif args.cmd == "api":
        return _cmd_api(args, as_json)
    return 0


def entry() -> None:
    sys.exit(main())
    return prof
