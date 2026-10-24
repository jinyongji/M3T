"""Adapt the reviewed v0.9 installer to the existing 900x600 v0.8 dataset."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "ef268f504c4acffe5ffafdde03b5f3bef46a6d6e5d9ab79710c26198aefb2167"
INSTALLER_RELATIVE = "中间结果文件/server_v09/installer.py"
OLD_NAME = "工业物体跟踪数据集-v0.8-900x600-20261016"
NEW_NAME = "工业物体跟踪数据集-v0.9-20261017"
CALIBRATION = (900, 600, [[1125.0, 0, 449.5], [0, 1125.0, 299.5], [0, 0, 1]])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patched_installer(original):
    if sha(original) != ORIGINAL_SHA256:
        raise ValueError("The bundled reference installer differs from the reviewed v0.9 input")
    text = original.decode("utf-8")
    replacements = [
        ("OLD_NAME='工业物体跟踪数据集-v0.8-20261015'", "OLD_NAME=" + repr(OLD_NAME)),
        ("!=(1800,1200,[[2250.,0,899.5],[0,2250.,599.5],[0,0,1.]])",
         "!=(900,600,[[1125.,0,449.5],[0,1125.,299.5],[0,0,1.]])"),
        ("Unexpected v0.8 camera calibration; no old data was changed",
         "Expected the existing v0.8 900x600 calibration; no old data was changed"),
        ("merged['sequence_revisions']={'camera_motion':'v0.8-20261015',MODE:REVISION}",
         "merged['sequence_revisions']={'camera_motion':'v0.8-900x600-20261016',MODE:REVISION}"),
        ("merged['resolution_by_motion_mode']={'camera_motion':[1800,1200],MODE:[900,600]}",
         "merged['resolution_by_motion_mode']={'camera_motion':[900,600],MODE:[900,600]}"),
    ]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError("Unexpected installer layout; no files changed: " + before)
        text = text.replace(before, after)
    return text.encode("utf-8")


def atomic_bytes(path, data):
    temporary = path.with_name(path.name + ".compat.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def apply(root, kit):
    root, kit = Path(root).resolve(), Path(kit).resolve()
    reference = (kit / "installer_v09_original.py").read_bytes()
    payload = patched_installer(reference)
    source = root / "交付物" / NEW_NAME
    target = root / "交付物" / OLD_NAME
    # Require the actual existing dataset before spending hours on rendering.
    for directory in (source, target):
        camera = json.loads((directory / "cameras/camera.json").read_text(encoding="utf-8-sig"))
        if (camera["width"], camera["height"], camera["K"]) != CALIBRATION:
            raise ValueError("Expected 900x600 camera with fx=fy=1125: " + str(directory))
    installer = root / INSTALLER_RELATIVE
    manifest = root / "SHA256SUMS-v0.9.txt"
    old_installer, old_manifest = installer.read_bytes(), manifest.read_bytes()
    if sha(old_installer) not in {ORIGINAL_SHA256, sha(payload)}:
        raise ValueError("The v0.9 installer has unknown edits; no files changed")
    lines = old_manifest.decode("utf-8-sig").splitlines()
    count = 0
    for index, line in enumerate(lines):
        digest, relative = line.split("  ", 1)
        if relative == INSTALLER_RELATIVE:
            if digest not in {ORIGINAL_SHA256, sha(payload)}:
                raise ValueError("Unexpected installer manifest entry")
            lines[index] = sha(payload) + "  " + relative
            count += 1
    if count != 1:
        raise ValueError("Expected exactly one installer entry in SHA256SUMS-v0.9.txt")
    new_manifest = ("\n".join(lines) + "\n").encode("utf-8")
    if old_installer == payload and old_manifest == new_manifest:
        print("900x600 compatibility patch already installed")
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root / "中间结果文件/compat_v09_backups" / stamp
    backup.mkdir(parents=True)
    (backup / "installer.py").write_bytes(old_installer)
    (backup / manifest.name).write_bytes(old_manifest)
    try:
        atomic_bytes(installer, payload)
        atomic_bytes(manifest, new_manifest)
    except BaseException:
        atomic_bytes(installer, old_installer)
        atomic_bytes(manifest, old_manifest)
        raise
    receipt = {"target": str(target), "source": str(source), "patched_installer_sha256": sha(payload),
               "resolution_by_motion_mode": {"camera_motion": [900, 600], "camera_and_objects_motion": [900, 600]},
               "backup": str(backup)}
    atomic_bytes(root / "中间结果文件/compat_v09_receipt.json",
                 (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print("900x600 compatibility patch installed. Target: " + str(target))
    print("Original installer and hash list: " + str(backup))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    apply(args.dataset_root, Path(__file__).parent)
