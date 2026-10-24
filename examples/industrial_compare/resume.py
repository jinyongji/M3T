"""Resume exactly the saved protocol; source/binary/data checks still run."""
import json
import os
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1]).resolve()
manifest = json.loads((output/"run_manifest.json").read_text(encoding="utf-8"))
if manifest["status"] == "complete":
    raise SystemExit("Run is already complete")
identity = manifest["identity"]
command = [sys.executable, "-u", str(Path(__file__).with_name("compare.py")), "--resume",
           "--output", str(output), "--dataset", identity["dataset"], "--cache", identity["cache"], "--frames", str(identity["frames"]),
           "--repeats", str(identity["repeats"]), "--warmup", str(identity["warmup"]), "--methods", *identity["methods"]]
if identity["disk_model"]:
    command += ["--disk-model", identity["disk_model"]]
for key in ("scene", "motion", "condition", "object"):
    for value in dict.fromkeys(c[key] for c in identity["configurations"]):
        command += ["--"+key, value]
with (output/"launcher.log").open("ab") as log, open(os.devnull, "rb") as devnull:
    process = subprocess.Popen(command, stdin=devnull, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(output/"launcher.pid").write_text(str(process.pid)+"\n", encoding="ascii")
print("Resume launched. PID: %d\nOutput: %s" % (process.pid, output))
