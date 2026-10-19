"""Start vm_guard.py as a detached process (it starts the workers). colab exec -s jianjin --env WORKERS=4 --env DEADLINE_H=9 -f vm_launch.py"""
import os
import subprocess
import sys

env = {**os.environ, "PYTHONUNBUFFERED": "1"}
p = subprocess.Popen([sys.executable, "/content/ds/vm_guard.py"], stdout=open("/content/logs/guard.out", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd="/content")
print("guard pid", p.pid, "WORKERS", env.get("WORKERS"), "DEADLINE_H", env.get("DEADLINE_H"))
