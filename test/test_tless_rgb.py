"""Baseline compatibility, experiment scope, support audits and paired metrics."""
import csv
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/tless_rgb'))
sys.path.insert(0,str(ROOT/'examples/tless'))
import rgb_build
import rgb_report
from rgb_run import select
import rgb_run
import tracking_data as data
import tracking_report as base
import test_tless_tracking as fixtures

VARIANTS=data.load(ROOT/'examples/tless_rgb/variants.json')

class DesignTests(unittest.TestCase):
    def test_generation_preserves_all_frozen_inputs_and_gates_new_observations(self):
        files=[ROOT/'examples/industrial_compare/baseline.cpp',ROOT/'src/texture_modality.cpp',ROOT/'include/m3t/texture_modality.h',ROOT/'examples/tless/tracking_driver.cpp']
        before={str(p):data.sha(p) for p in files}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);rgb_build.generate(files[0],ROOT,out,ROOT/'examples/tless')
            code=(out/'tless.cpp').read_text()
            self.assertIn('std::make_shared<tless_rgb::VerifiedTexture>',code)
            self.assertIn('in.video_verify=in.variant=="B1"||in.variant=="B3"',code)
            self.assertIn('region->set_use_visibility_consensus(input_.visibility_consensus)',code)
            self.assertLess(code.index('const int rgb_candidates='),code.index('if(!success){const auto reset_start='))
            self.assertIn('times.results<<\',\'<<rgb_candidates',code)
            self.assertIn('body->has_texture()',code)
            self.assertEqual((out/'override/m3t/texture_modality.h').read_text().count('RgbAdapterPoints()'),1)
        self.assertEqual(before,{str(p):data.sha(p) for p in files})

    def test_fixed_pilot_is_selected_by_scene_and_angle_not_success_or_visibility(self):
        counts=[4,3,5,4,5,5,8,6,12,6,4,6,8,7,6,8,7,7,9,18]
        configs=[dict(scene_id=s,elevation=e,gt_index=i,id=f'{s}/{e}/{i}') for s,n in enumerate(counts,1) for e in data.PROTOCOL['elevations'] for i in range(n)]
        index=dict(configurations=configs)
        smoke,frames=select(index,'smoke');self.assertEqual((len(smoke),frames),(3,21))
        pilot,frames=select(index,'pilot');self.assertEqual((len(pilot),frames),(63,72))
        self.assertEqual(len(pilot)*len(VARIANTS)*(frames-1),22365)
        full,frames=select(index,'full');self.assertEqual((len(full),frames),(966,72))
        with self.assertRaises(ValueError):select(dict(configurations=configs[:-1]),'full')

class AuditTests(unittest.TestCase):
    def fixture(self,variant):
        helper=fixtures.OutputAuditTests();self.addCleanup(helper.doCleanups)
        root,prepared,index,config,old,rows=helper.fixture();folder=root/'ours';folder.mkdir()
        meta=data.load(old/'metadata.json');meta.update(method='ours',cad_disk_configured=1,rgb_variant=variant['id'],
            rgb_video_verification=int(variant['video_verify']),rgb_visibility_consensus=int(variant['visibility']),rgb_region_control=int(variant['region_control']))
        data.save(folder/'metadata.json',meta)
        for row in rows:row.update(rgb_video_candidates=10,rgb_video_retained=0 if variant['region_control'] else 7 if variant['video_verify'] else 10,
            rgb_video_verified=int(variant['video_verify']),rgb_visibility_evidence=.8,rgb_visibility_activation=.2 if variant['visibility'] else 0)
        return helper,root,prepared,index,config,folder,rows

    def test_new_flags_retain_original_pre_reset_pose_and_decode_audit(self):
        for variant in VARIANTS:
            h,root,p,index,c,folder,rows=self.fixture(variant);h.write(folder,rows)
            r,e,t=rgb_report.analyze(folder,c,index,p,root,3,variant,{},{},data,base)
            self.assertEqual((r['successes'],r['resets']),(1,1))
            self.assertEqual(t[:,base.TIMES.index('compute_wall_us')].tolist(),[5,5])

    def test_invalid_native_geometric_support_is_rejected(self):
        v=VARIANTS[1];h,root,p,index,c,folder,rows=self.fixture(v)
        rows[0]['rgb_video_retained']=2;h.write(folder,rows)
        with self.assertRaisesRegex(ValueError,'Insufficient verified support'):rgb_report.analyze(folder,c,index,p,root,3,v,{},{},data,base)

    def test_claimed_rejection_cannot_retain_video_constraints(self):
        v=VARIANTS[1];h,root,p,index,c,folder,rows=self.fixture(v)
        rows[0]['rgb_video_verified']=0;h.write(folder,rows)
        with self.assertRaisesRegex(ValueError,'Rejected video'):rgb_report.analyze(folder,c,index,p,root,3,v,{},{},data,base)

    def test_runtime_variant_mismatch_is_rejected(self):
        v=VARIANTS[0];h,root,p,index,c,folder,rows=self.fixture(v);h.write(folder,rows)
        with self.assertRaisesRegex(ValueError,'runtime flags'):rgb_report.analyze(folder,c,index,p,root,3,VARIANTS[3],{},{},data,base)

    def test_regression_ignores_timing_but_checks_pose_and_reset_segment(self):
        v=VARIANTS[0];h,root,p,index,c,left,rows=self.fixture(v);h.write(left,rows)
        right=root/'reference';right.mkdir();other=[dict(r,compute_wall_us=999999) for r in rows];h.write(right,other)
        self.assertTrue(rgb_report.regression(left,right,base)['passed'])
        other[0]['est_tx']=.03;h.write(right,other)
        result=rgb_report.regression(left,right,base);self.assertFalse(result['passed']);self.assertEqual(result['first_difference']['row_index'],0)
        other=[dict(r) for r in rows];other[1]['segment_id']=0;h.write(right,other)
        self.assertFalse(rgb_report.regression(left,right,base)['passed'])

    def test_paired_summary_keeps_invalid_and_low_visibility_denominators(self):
        helper=fixtures.SummaryTests();records,index=helper.fixture();converted=[]
        for v,(r,e,t) in zip(VARIANTS,records):
            r=dict(r,variant=v['id'],rgb_geometry=dict(candidates=0,retained=0,verified_frames=0,rejected_frames=0))
            converted.append((r,np.asarray(e,dtype=float),np.asarray([[x[k] for k in base.TIMES] for x in t])))
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);rgb_report.summarize(converted,VARIANTS,out,'pilot',{},data,base)
            value=data.load(out/'summary.json')['variants']['B3'];self.assertEqual(value['success_percent'],90)
            self.assertEqual(value['timing']['compute_wall_us']['mean'],1000)
            self.assertEqual(value['timing']['compute_wall_us']['frames'],10)
            self.assertEqual(value['visibility']['under_10_percent']['frames'],1)
            self.assertEqual(value['invalid'],1)
            with self.assertRaisesRegex(ValueError,'Incomplete paired'):rgb_report.summarize(converted[:-1],VARIANTS,out,'pilot',{},data,base)

class RunnerTests(unittest.TestCase):
    def exercise(self,passed):
        import contextlib
        import tracking_run
        import tracking_build
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        configs=[dict(id=f'scene_{i:02d}',scene_id=i,obj_id=1,elevation=75,gt_index=0) for i in [1,6,20]]
        index=dict(configurations=configs,fingerprint='synthetic',protocol=data.PROTOCOL)
        a=SimpleNamespace(output=root/'smoke',stage='smoke',resume=False,smoke_run=None,dataset=root/'dataset',cache=root/'cache',
            kit=ROOT/'examples/industrial_compare',tless_kit=ROOT/'examples/tless',disk_model=root/'disk.onnx')
        calls=[];real_sha=data.sha
        def fake_native(command,folder,module):folder.mkdir(parents=True,exist_ok=True);calls.append(command)
        def fake_analyze(folder,c,index,prepared,dataset,frames,v,*unused):
            r=dict(config_id=c['id'],variant=v['id'],frames=20,successes=18,resets=2,invalid=0,audit_disagreements=[],
                rgb_geometry=dict(candidates=20,retained=10,verified_frames=1,rejected_frames=0))
            e=np.asarray([[.05,3,.8,1,0]]*18+[[.2,20,.05,1,0]]*2)
            t=np.zeros((20,len(base.TIMES)));t[:,base.TIMES.index('compute_wall_us')]=1000
            return r,e,t
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict('os.environ',CUDA_VISIBLE_DEVICES='GPU-test'))
            stack.enter_context(patch.object(data,'sha',side_effect=lambda p:'5ff1ecfd9a31009a12d0e73ec76b476291756fe101b0417e3d0cc90a0dd42648' if Path(p)==a.disk_model else real_sha(p)))
            stack.enter_context(patch.object(data,'prepare',return_value=(root/'prepared',index)))
            stack.enter_context(patch.object(data,'rgb_preflight',return_value={'headers':'fixed'}))
            stack.enter_context(patch.object(tracking_run,'lock'))
            stack.enter_context(patch.object(tracking_build,'verified',return_value={'binary_sha256':'a'*64}))
            stack.enter_context(patch.object(rgb_run,'verified',return_value={'binary_sha256':'b'*64}))
            stack.enter_context(patch.object(rgb_run,'select',return_value=(configs,21)))
            stack.enter_context(patch.object(rgb_run,'native',side_effect=fake_native))
            stack.enter_context(patch.object(rgb_run,'analyze',side_effect=fake_analyze))
            stack.enter_context(patch.object(base,'analyze'))
            stack.enter_context(patch.object(rgb_run,'regression',return_value={'passed':passed,'frames':20}))
            if passed:
                rgb_run.execute(a);self.assertEqual(len(calls),18)
                self.assertEqual(data.load(a.output/'run_manifest.json')['status'],'complete')
                a.resume=True;rgb_run.execute(a);self.assertEqual(len(calls),18)
                self.assertTrue(data.load(a.output/'regression_gate.json')['passed'])
            else:
                with self.assertRaisesRegex(ValueError,'B0 differs'):rgb_run.execute(a)
                self.assertEqual(data.load(a.output/'run_manifest.json')['status'],'failed')
                self.assertFalse(data.load(a.output/'regression_gate.json')['passed'])
                a.smoke_run=a.output;a.output=root/'pilot';a.stage='pilot'
                with self.assertRaisesRegex(ValueError,'smoke regression failed'):rgb_run.execute(a)

    def test_completed_smoke_resume_reaudits_without_reexecuting_jobs(self):self.exercise(True)
    def test_failed_baseline_regression_blocks_pilot(self):self.exercise(False)

if __name__=='__main__':unittest.main()
