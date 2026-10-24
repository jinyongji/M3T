"""Synthetic replacement/rollback tests; these do not render or claim a GPU run."""
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "examples/industrial_render_v09"
sys.path.insert(0, str(KIT))
import apply_compat

REFERENCE = (KIT / "installer_v09_original.py").read_bytes()
PAYLOAD = apply_compat.patched_installer(REFERENCE)
installer = types.ModuleType("patched_v09_installer")
exec(compile(PAYLOAD, "patched_v09_installer.py", "exec"), installer.__dict__)
# Blender's real Python supports file_digest. The local Python 3.8 test runtime
# uses an equivalent hash helper on these tiny synthetic files.
installer.sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def json_write(path, content):
    write(path, (json.dumps(content, ensure_ascii=False) + "\n").encode("utf-8"))


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / ".v09-checks"
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="case_", dir=str(base))
        self.root = Path(self.temporary.name).resolve()
        self.root.relative_to(base.resolve())
        self.old = self.root / "交付物" / apply_compat.OLD_NAME
        self.new = self.root / "交付物" / apply_compat.NEW_NAME
        self.work = self.root / "中间结果文件/run_state_v09"
        self.work.mkdir(parents=True)
        self.camera = {"width":900,"height":600,"K":apply_compat.CALIBRATION[2],"fps":30,
                       "frames_per_sequence":500,"distortion_coefficients":[0]*5,"translation_unit":"meter",
                       "pose_direction":"p_camera = R @ p_object + t",
                       "motion_modes":["camera_motion",installer.MODE],
                       "conditions":["01_normal","02_low_light","03_hard_glare","04_fast_motion","05_sensor_noise"],
                       "models":[{"object_id":i,"units":"meter"} for i in range(1,16)]}
        for directory in (self.old, self.new):
            camera = dict(self.camera)
            if directory == self.new:camera["motion_modes"] = [installer.MODE]
            json_write(directory / "cameras/camera.json", camera)
            write(directory / "cameras/K.txt", b"1125 0 449.5\n0 1125 299.5\n0 0 1\n")
            write(directory / "objects/obj_000001/obj_000001.obj", b"same geometry and texture references")
        for sid in range(1,6):
            scene = "scene_%02d" % sid
            for category in ("images","videos","poses"):
                write(self.old / category / scene / "camera_motion/original.txt", b"preserved camera-only output")
                for condition in self.camera["conditions"]:
                    suffix = condition + ".mp4" if category == "videos" else condition + "/frame.txt"
                    write(self.old / category / scene / installer.MODE / suffix, b"old moving output")
                    write(self.new / category / scene / installer.MODE / suffix, b"new moving output")
            write(self.old / "scenes" / scene / (scene+"-camera_motion-v0.8-20261015.blend"), b"camera-only scene")
            write(self.old / "scenes" / scene / (scene+"-"+installer.MODE+"-v0.8-20261015.blend"), b"old moving scene")
            write(self.new / "scenes" / scene / (scene+"-"+installer.MODE+"-v0.9-20261017.blend"), b"new moving scene")
        self.audit = {"status":"passed","scope":"complete dataset","images":12500,"videos":25,
                      "frames_per_video":500,"pose_rows":37500,"resolution":[900,600]}
        json_write(self.work / "validation/final_audit.json", self.audit)
        self.old_manifest = self.root / "SHA256SUMS.txt"
        lines = [installer.sha(path)+"  "+path.relative_to(self.root).as_posix()
                 for path in sorted(self.old.rglob("*")) if path.is_file()]
        write(self.old_manifest, ("\n".join(lines)+"\n").encode("utf-8"))
        self.before = snapshot(self.old)
        self.before_manifest = self.old_manifest.read_bytes()

    def tearDown(self):
        # The generated target was verified to remain inside the test workspace.
        self.root.relative_to((ROOT / ".v09-checks").resolve())
        self.temporary.cleanup()

    def verify_unchanged(self):
        self.assertEqual(snapshot(self.old), self.before)
        self.assertEqual(self.old_manifest.read_bytes(), self.before_manifest)

    def install(self):
        return installer.install_completed_dataset(self.root,self.new,self.work)

    def test_patch_is_guarded_and_idempotent(self):
        target = self.root / apply_compat.INSTALLER_RELATIVE
        manifest = self.root / "SHA256SUMS-v0.9.txt"
        write(target, REFERENCE)
        write(manifest, (apply_compat.ORIGINAL_SHA256+"  "+apply_compat.INSTALLER_RELATIVE+"\n").encode("utf-8"))
        apply_compat.apply(self.root,KIT)
        self.assertEqual(target.read_bytes(),PAYLOAD)
        self.assertIn(apply_compat.sha(PAYLOAD),manifest.read_text(encoding="utf-8"))
        backups = list((self.root / "中间结果文件/compat_v09_backups").iterdir())
        apply_compat.apply(self.root,KIT)
        self.assertEqual(list((self.root / "中间结果文件/compat_v09_backups").iterdir()),backups)
        self.verify_unchanged()

    def test_unknown_installer_edit_rejected_before_write(self):
        target = self.root / apply_compat.INSTALLER_RELATIVE
        manifest = self.root / "SHA256SUMS-v0.9.txt"
        write(target,REFERENCE+b"\n# unknown edits\n")
        original = (apply_compat.ORIGINAL_SHA256+"  "+apply_compat.INSTALLER_RELATIVE+"\n").encode("utf-8")
        write(manifest,original)
        with self.assertRaisesRegex(ValueError,"unknown edits"):
            apply_compat.apply(self.root,KIT)
        self.assertEqual(manifest.read_bytes(),original)
        self.verify_unchanged()

    def test_replacement_preserves_camera_only_and_uniform_900_calibration(self):
        result = self.install()
        self.assertEqual(result["status"],"installed")
        self.assertEqual(result["replacement_units"],23)
        for relative,data in self.before.items():
            if "/camera_motion/" in relative or "-camera_motion-" in relative or relative.startswith("objects/"):
                self.assertEqual((self.old / relative).read_bytes(),data)
        merged = json.loads((self.old / "cameras/camera.json").read_text(encoding="utf-8"))
        self.assertEqual(merged["K"],self.camera["K"])
        self.assertEqual(merged["resolution_by_motion_mode"],{"camera_motion":[900,600],installer.MODE:[900,600]})
        backup = Path(result["backup"])
        for relative,data in self.before.items():
            if "/camera_and_objects_motion/" in relative or "-camera_and_objects_motion-" in relative or relative=="cameras/camera.json":
                saved = backup / self.old.relative_to(self.root) / relative
                self.assertEqual(saved.read_bytes(),data)
        for line in self.old_manifest.read_text(encoding="utf-8").splitlines():
            digest,relative = line.split("  ",1)
            self.assertEqual(digest,installer.sha(self.root / relative))
        sys.path.insert(0,str(ROOT / "examples/industrial"))
        from dataset import read_camera
        self.assertEqual(read_camera(self.old)["width"],900)

    def test_second_replacement_is_noop(self):
        self.install()
        before = snapshot(self.old)
        self.assertEqual(self.install()["status"],"already_installed")
        self.assertEqual(snapshot(self.old),before)

    def test_incomplete_render_rejected(self):
        json_write(self.work / "validation/final_audit.json",dict(self.audit,images=15,scope="smoke test only"))
        with self.assertRaisesRegex(RuntimeError,"completely rendered"):
            self.install()
        self.verify_unchanged()

    def test_mismatched_models_rejected(self):
        write(self.new / "objects/obj_000001/obj_000001.obj",b"different object")
        with self.assertRaisesRegex(RuntimeError,"models differ"):
            self.install()
        self.verify_unchanged()

    def test_mismatched_calibration_rejected(self):
        json_write(self.new / "cameras/camera.json",dict(self.camera,width=1800))
        with self.assertRaisesRegex(RuntimeError,"v0.9 camera"):
            self.install()
        self.verify_unchanged()

    def test_mismatched_old_manifest_rejected(self):
        write(self.old / "images/scene_01/camera_motion/original.txt",b"unrecorded old input change")
        changed = snapshot(self.old)
        with self.assertRaisesRegex(RuntimeError,"differs from its manifest"):
            self.install()
        self.assertEqual(snapshot(self.old),changed)
        self.assertEqual(self.old_manifest.read_bytes(),self.before_manifest)

    def test_faults_rollback_after_backup_and_after_publish(self):
        original = installer.commit_transaction
        for index,phase in [(0,"after_backup"),(0,"after_publish"),(1,"after_backup"),(1,"after_publish"),(21,"after_publish"),(22,"after_publish")]:
            def fail(number,stage):
                if number==index and stage==phase:raise RuntimeError("injected failure")
            def commit(root,work,entries,backup):
                return original(root,work,entries,backup,hook=fail)
            installer.commit_transaction = commit
            try:
                with self.assertRaisesRegex(RuntimeError,"injected failure"):
                    self.install()
            finally:
                installer.commit_transaction = original
            self.verify_unchanged()

    def test_recover_interrupted_journal(self):
        destination = self.old / "images/scene_01" / installer.MODE
        backup = self.root / "交付物-backup/simulated" / destination.relative_to(self.root)
        ready = self.work / "staged/ready"
        entry = {"destination":str(destination),"backup":str(backup),"prepared":str(ready),"had_previous":True}
        backup.parent.mkdir(parents=True)
        destination.replace(backup)
        write(destination / "new.txt",b"interrupted new output")
        json_write(self.work / "install_transaction.json",{"id":"simulated","phase":"applying","entries":[entry]})
        installer.rollback(self.work / "install_transaction.json",self.root,self.work)
        self.verify_unchanged()


if __name__ == "__main__":
    unittest.main()
