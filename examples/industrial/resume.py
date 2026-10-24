"""Reconstruct an interrupted run without changing its selection or binary."""
import json
import os
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1]).resolve()
manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
identity = manifest["identity"]
if manifest["status"] == "complete":
    raise SystemExit("Run is already complete")
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("Select your allocated GPU before resuming")
command = [sys.executable, "-u", str(Path(__file__).with_name("run.py")), "--resume",
           "--source", os.environ.get("M3T_SOURCE", "/home/jyj/Projects/M3T"),
           "--dataset", identity["dataset"], "--frames", str(identity["frames_including_initialization"]),
           "--output", str(output)]
for key in ("scene", "motion", "condition", "object"):
    for value in dict.fromkeys(config[key] for config in identity["configurations"]):
        command.extend(["--"+key, value])
with (output / "launcher.log").open("ab") as log, open(os.devnull, "rb") as devnull:
    process = subprocess.Popen(command, stdin=devnull, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(output / "launcher.pid").write_text(str(process.pid)+"\n", encoding="ascii")
print("Resume launched. PID: %d\nOutput: %s" % (process.pid, output))
