"""Start the training workers, watch them, enforce the budget, release the VM.

Runs detached on the VM (started by vm_launch.py), not inside a long ``colab exec``: on 2026-10-19 the
long exec's websocket dropped at 12:33 and the VM was reclaimed about 35 minutes later, so liveness now
comes from short local execs every 5 minutes (poll.sh) and this loop only does the bookkeeping.
DEADLINE_H counts hours from the moment
vm_setup.py ran (the VM's first minutes): past it the workers are killed, whatever has finished
stays in /content/results. When the workers are done or killed, it waits up to 60 minutes for
the local poller to download /content/snap.tgz and write /content/results/PULLED, then unassigns the VM
so compute units stop being billed.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

R = Path("/content/results")
LOGS = Path("/content/logs")
LOGS.mkdir(exist_ok=True)
R.mkdir(exist_ok=True)
start = Path("/content/ds/vm_start").stat().st_mtime
deadline = start + float(os.environ.get("DEADLINE_H", "7")) * 3600
n_workers = int(os.environ.get("WORKERS", "1"))


def say(*parts):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in parts)
    print(line, flush=True)
    with open(LOGS / "guard.log", "a") as f:
        f.write(line + "\n")


def snapshot():
    # one file for the local poller to download: results and logs so far
    subprocess.run("tar czf /content/snap.tgz.part -C /content results logs && mv /content/snap.tgz.part /content/snap.tgz",
                   shell=True)


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split()[2] != "Z"


pidfile = Path("/content/ds/pids.json")
if pidfile.exists():
    pids = json.loads(pidfile.read_text())
else:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1" if n_workers > 1 else "2", "MUJOCO_GL": "egl"}
    pids = {f"w{i}": subprocess.Popen([sys.executable, "/content/ds/vm_runner.py", "main", f"w{i}"],
                                      stdout=open(LOGS / f"worker_w{i}.log", "a"), stderr=subprocess.STDOUT,
                                      start_new_session=True, env=env, cwd="/content").pid
            for i in range(n_workers)}
    pidfile.write_text(json.dumps(pids))
    say("started", pids, "deadline", time.strftime("%H:%M", time.localtime(deadline)))

stopped = False
while True:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    done = sorted(p.stem for p in R.glob("square_*.json"))
    running = [w for w, pid in pids.items() if alive(pid)]
    say(f"{(time.time() - start) / 3600:.2f} h | results {len(done)}/12 | workers alive {running} | gpu {gpu}")
    snapshot()
    if not running:
        break
    if time.time() > deadline:
        for pid in pids.values():
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
        (R / "STOPPED_BUDGET").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        say("deadline reached, workers killed")
        stopped = True
        break
    time.sleep(300)

(R / "ALL_DONE").write_text("stopped at deadline" if stopped else "finished")
snapshot()
waited = 0
while not (R / "PULLED").exists() and waited < 3600:
    time.sleep(30)
    waited += 30
say("results pulled" if (R / "PULLED").exists() else "no PULLED marker after 60 min", "; releasing the VM")
time.sleep(5)
if os.environ.get("KEEP_VM") != "1":
    try:   # may not work outside the kernel; the local poller stops the VM as well
        from google.colab import runtime
        runtime.unassign()
    except Exception as exc:
        say("unassign from the guard failed:", exc)
