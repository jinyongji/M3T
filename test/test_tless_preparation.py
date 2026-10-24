"""T-LESS preparation must preserve paths and reject a partial test split."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tless_preparation', ROOT/'examples/tless/prepare_tless.py')
tless = importlib.util.module_from_spec(spec); spec.loader.exec_module(tless)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.data = self.root/'tless'; self.data.mkdir()

    def test_base_prefix_stripped_and_model_relative_path_preserved(self):
        archive = self.root/'input.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('tless/camera_primesense.json', '{}')
            z.writestr('models_cad/obj_000001.ply', 'ply\nformat ascii 1.0\nend_header\n')
        with patch.object(tless.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)):
            self.assertEqual(tless.extract_archive(archive, self.data), 2)
        self.assertEqual((self.data/'camera_primesense.json').read_text(), '{}')
        self.assertTrue((self.data/'models_cad/obj_000001.ply').is_file())
        self.assertFalse((self.data/'tless').exists())

    def test_traversal_or_symlink_archive_rejected_before_writes(self):
        for index, name in enumerate(['../escape', '/absolute', 'test/../../escape']):
            archive = self.root/('bad%d.zip' % index)
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr(name, 'bad')
            with self.assertRaises(ValueError):
                tless.extract_archive(archive, self.data)
        archive = self.root/'link.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            info = zipfile.ZipInfo('test/link'); info.external_attr = 0o120777 << 16
            z.writestr(info, '../../outside')
        with self.assertRaises(ValueError):
            tless.extract_archive(archive, self.data)
        self.assertEqual(list(self.data.iterdir()), [])

    def test_partial_scene_collection_cannot_be_called_complete(self):
        models = self.data/'models_cad'; models.mkdir()
        tless.save(models/'models_info.json', {str(i): {'diameter': 50} for i in range(1,31)})
        for i in range(1,31):
            (models/('obj_%06d.ply' % i)).write_text('ply\nformat ascii 1.0\nend_header\n')
        (self.data/'test_primesense/000001').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'all 20'):
            tless.audit(self.data, self.root/'audit.json')
        self.assertFalse((self.root/'audit.json').exists())

    def test_bad_rigid_rotation_not_accepted_as_pose(self):
        self.assertTrue(tless.valid_rotation([1,0,0,0,1,0,0,0,1]))
        self.assertFalse(tless.valid_rotation([-1,0,0,0,1,0,0,0,1]))
        self.assertFalse(tless.valid_rotation([2,0,0,0,1,0,0,0,1]))
        self.assertFalse(tless.valid_rotation([float('nan')]+[0]*8))

    def test_mirror_resume_keeps_revision_checksum_and_official_source(self):
        name = 'tless_base.zip'; payload = b'pinned archive bytes'
        digest = hashlib.sha256(payload).hexdigest()
        archives = self.root/'tless_archives'; archives.mkdir()
        partial = archives/(name+'.partial'); partial.write_bytes(payload[:6])

        def curl(command, **kwargs):
            self.assertEqual(command[-1], 'https://hf-mirror.com/datasets/bop-benchmark/tless/resolve/'+tless.REVISION+'/'+name)
            self.assertEqual(command[command.index('--continue-at')+1], '-')
            self.assertEqual(Path(command[command.index('--output')+1]), partial)
            self.assertEqual(partial.read_bytes(), payload[:6])
            partial.write_bytes(payload)

        with patch.dict(tless.os.environ, {'HF_ENDPOINT': 'https://hf-mirror.com/'}), \
                patch.dict(tless.FILES, {name: (len(payload), digest)}, clear=True), \
                patch.object(tless.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)), \
                patch.object(tless.subprocess, 'run', side_effect=curl) as run, \
                patch.object(tless, 'extract_archive', return_value=3) as extract:
            tless.download(self.data)
        run.assert_called_once(); extract.assert_called_once_with(archives/name, self.data)
        self.assertEqual((archives/name).read_bytes(), payload)
        manifest = json.loads((self.data/'download_manifest.json').read_text())
        self.assertEqual(manifest['source'], tless.SOURCE)
        self.assertEqual(manifest['download_endpoint'], 'https://hf-mirror.com')
        self.assertEqual(manifest['files'][name]['sha256'], digest)

    def test_completed_partial_verified_without_redownload(self):
        name = 'tless_base.zip'; payload = b'completed archive'
        archives = self.root/'tless_archives'; archives.mkdir()
        (archives/(name+'.partial')).write_bytes(payload)
        with patch.dict(tless.FILES, {name: (len(payload), hashlib.sha256(payload).hexdigest())}, clear=True), \
                patch.object(tless.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)), \
                patch.object(tless.subprocess, 'run') as run, \
                patch.object(tless, 'extract_archive', return_value=3):
            tless.download(self.data, 'https://hf-mirror.com')
        run.assert_not_called()
        self.assertEqual((archives/name).read_bytes(), payload)

    def test_mirror_payload_with_wrong_checksum_not_extracted(self):
        name = 'tless_base.zip'; payload = b'expected archive'
        archives = self.root/'tless_archives'; archives.mkdir()
        (archives/(name+'.partial')).write_bytes(b'X'*len(payload))
        with patch.dict(tless.FILES, {name: (len(payload), hashlib.sha256(payload).hexdigest())}, clear=True), \
                patch.object(tless.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)), \
                patch.object(tless, 'extract_archive') as extract:
            with self.assertRaisesRegex(ValueError, 'size/SHA256'):
                tless.download(self.data, 'https://hf-mirror.com')
        extract.assert_not_called()
        self.assertFalse((archives/name).exists())
        self.assertFalse((self.data/'download_manifest.json').exists())

    def test_endpoint_explicit_override_and_invalid_values(self):
        with patch.dict(tless.os.environ, {'HF_ENDPOINT': 'https://hf-mirror.com'}):
            self.assertEqual(tless.download_endpoint('https://huggingface.co/'), 'https://huggingface.co')
            self.assertEqual(tless.download_endpoint(), 'https://hf-mirror.com')
        for endpoint in ['', 'file:///tmp/mirror', 'https://host/path', 'https://user:password@host', 'https://host?token=abc']:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                tless.download_endpoint(endpoint)


if __name__ == '__main__':
    unittest.main()
