"""Runs the whole Smart Traffic Violation system from one command.

    .venv\\Scripts\\python.exe run_system.py              receiver + AI pipeline + dashboard
    .venv\\Scripts\\python.exe run_system.py --no-dashboard
    .venv\\Scripts\\python.exe run_system.py --check       only run the pre-flight checks

What it starts:
  1. the receiver  (aiot_project/receiver/cam_receiver_code.py): listens on TCP 5000 for the DevKit's
     CAPTURE messages, fetches a snapshot from the ESP32-CAM, and hands it to the AI pipeline
     (Traffic-Violation-main/pipeline/aggregator.py -> violations.db + violations/evidence/)
  2. the dashboard (Traffic-Violation-main/dashboard/app.py): http://localhost:8501

Hardware (DevKit + ESP32-CAM) must be powered and on the same Wi-Fi as this PC for real triggers.
Ctrl+C stops everything.
"""
import argparse
import os
import re
import socket
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AI_DIR = ROOT / "Traffic-Violation-main"
IOT_DIR = ROOT / "aiot_project"
RECEIVER = IOT_DIR / "receiver" / "cam_receiver_code.py"
DEVKIT_SKETCH = IOT_DIR / "firmware" / "trafficViolationSystem" / "trafficViolationSystem.ino"
DASHBOARD = AI_DIR / "dashboard" / "app.py"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

WEIGHTS = [
    AI_DIR / "training" / "runs" / "yolo26" / "helmet_plate" / "weights" / "best.pt",
    AI_DIR / "training" / "runs" / "yolo26" / "triple_riding" / "weights" / "best.pt",
]
CLASS_NAME_FILES = [
    AI_DIR / "dataset" / "helmet_and_plate" / "data.yaml",
    AI_DIR / "dataset" / "triple_riding" / "data.yaml",
]
RECEIVER_PORT = 5000
DASHBOARD_PORT = 8501


def pc_ip():
    """The address the DevKit should use to reach this PC (the interface with a default route)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def find_value(path, pattern):
    match = re.search(pattern, Path(path).read_text(encoding="utf-8", errors="ignore"))
    return match.group(1) if match else None


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def preflight():
    """Returns (errors, warnings). Errors stop the launch; warnings are only printed."""
    errors, warnings = [], []

    if not VENV_PYTHON.exists():
        errors.append(f"virtual environment not found: {VENV_PYTHON}")
    for path in WEIGHTS + CLASS_NAME_FILES:
        if not path.exists():
            errors.append(f"missing: {path}")
    if port_in_use(RECEIVER_PORT):
        errors.append(f"TCP port {RECEIVER_PORT} is already in use (is the receiver already running?)")

    ip = pc_ip()
    devkit_target = find_value(DEVKIT_SKETCH, r'PYTHON_SERVER_IP\s*=\s*"([^"]+)"')
    camera_ip = find_value(RECEIVER, r'CAMERA_IP\s*=\s*"([^"]+)"')
    print(f"  this PC's IP        : {ip}")
    print(f"  DevKit sends to     : {devkit_target}   (PYTHON_SERVER_IP in the sketch)")
    print(f"  camera expected at  : {camera_ip}   (CAMERA_IP in the receiver)")
    if ip and devkit_target and ip != devkit_target:
        warnings.append(
            f"the DevKit firmware sends to {devkit_target} but this PC is {ip}: "
            "update PYTHON_SERVER_IP and re-flash the DevKit"
        )
    if camera_ip:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            if s.connect_ex((camera_ip, 80)) != 0:
                warnings.append(f"camera not reachable at {camera_ip}:80 (powered off / different Wi-Fi / new IP)")

    return errors, warnings


def stream(proc, tag):
    for line in proc.stdout:
        print(f"[{tag}] {line}", end="")


def main():
    parser = argparse.ArgumentParser(description="Run the whole traffic violation system.")
    parser.add_argument("--no-dashboard", action="store_true", help="start only the receiver + AI pipeline")
    parser.add_argument("--check", action="store_true", help="run the pre-flight checks and exit")
    args = parser.parse_args()

    print("Pre-flight checks:")
    errors, warnings = preflight()
    for w in warnings:
        print(f"  WARNING: {w}")
    for e in errors:
        print(f"  ERROR:   {e}")
    if errors:
        raise SystemExit("Fix the errors above and run again.")
    if args.check:
        print("Checks passed." if not warnings else "Checks passed with warnings.")
        return

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    commands = [("receiver", [str(VENV_PYTHON), "-u", str(RECEIVER)], IOT_DIR / "receiver")]
    if not args.no_dashboard:
        commands.append((
            "dashboard",
            [str(VENV_PYTHON), "-m", "streamlit", "run", str(DASHBOARD),
             "--server.headless", "true", "--server.port", str(DASHBOARD_PORT)],
            AI_DIR,
        ))

    procs = []
    try:
        for tag, cmd, cwd in commands:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            procs.append((tag, proc))
            threading.Thread(target=stream, args=(proc, tag), daemon=True).start()

        if not args.no_dashboard:
            print(f"\nDashboard: http://localhost:{DASHBOARD_PORT}")
        print("Running. Press Ctrl+C to stop.\n")

        while all(p.poll() is None for _, p in procs):
            threading.Event().wait(1)
        for tag, p in procs:
            if p.poll() is not None:
                print(f"\n[{tag}] exited with code {p.returncode}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
