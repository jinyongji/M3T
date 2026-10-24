"""Controlled switches, original-source preservation, pre-reset auditing and pairing."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/ablation'))
import ablation_data as data
from ablation_build import generate
from ablation_audit import TIMES,analyze,regression
from ablation_report import summarize
from ablation_run import model_cache_key


class DesignTests(unittest.TestCase):
    def test_nine_independent_groups_and_total_scope(self):
        variants=data.variants();flags={v['id']:v['flags'] for v in variants}
        self.assertEqual(flags,dict(A0='00000',A1='11111',A2='01111',A3='10111',A4='11011',A5='11101',A6='11110',A7='11001',A8='00001'))
        self.assertEqual(sum(v['configurations']*9 for v in data.LANES.values()),4914)
        self.assertEqual(sum(v['configurations']*9*(v['frames']-1) for v in data.LANES.values()),2776734)
        self.assertEqual([data.LANES[x]['gpu'] for x in data.LANES],[1,4,7])

    def test_generated_code_decouples_prior_without_changing_frozen_sources(self):
        source=ROOT/'examples/rbot_evaluator.cpp';before=source.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            generate(ROOT,Path(tmp));cpp=(Path(tmp)/'rbot_evaluator.cpp').read_text();cli=(Path(tmp)/'ablation.cpp').read_text()
            self.assertIn('use_ablation_prediction_smoothing_ &&\n                 smoothed_prediction_ready',cpp)
            self.assertIn('if (use_constant_velocity_prediction_)',cpp)
            self.assertIn('if (use_directional_temporal_prior_ ||',cpp)
            cad=cli.split('if (cad) {',1)[1].split('t->set_descriptor_distance_threshold',1)[0]
            yes,no=cad.split('} else {',1)
            self.assertIn('FocusedTexturedRenderer',yes);self.assertNotIn('FocusedTexturedRenderer',no)
            self.assertIn('DoNotUseModelTextureMatching()',no);self.assertIn('set_use_model_texture_disk(false)',no)
            self.assertIn('set_use_directional_temporal_prior(prior)',cli)
            self.assertIn('set_use_constant_velocity_prediction(prediction)',cli)
            self.assertIn('set_use_noise_adaptive_filtering(noise)',cli)
            self.assertEqual(source.read_bytes(),before)
            texture=(Path(tmp)/'texture_modality.cpp').read_text()
            self.assertIn('if (!data_point.from_model_texture && !use_model_texture_matching_)',texture)
            self.assertIn('gradient_ -= (weight * diff.transpose()) * dx_dtheta;',texture)
            self.assertIn('(weight * dx_dtheta.transpose()) * dx_dtheta;',texture)
            self.assertTrue((Path(tmp)/'video_only_arithmetic.patch').is_file())

    def test_template_reuse_requires_unchanged_engine_and_evaluator(self):
        old=dict(binary_sha256='a'*64,sources={'ours':'same frozen engine'},helpers={
            'build_inputs/rbot_evaluator.cpp':'same cpp','build_inputs/rbot_evaluator.h':'same header'})
        current=dict(old,binary_sha256='b'*64)
        prior=dict(identity=dict(build=old))
        self.assertEqual(model_cache_key(current,prior),'a'*16)
        changed=dict(current,sources={'ours':'changed engine'})
        with self.assertRaisesRegex(ValueError,'engines'):model_cache_key(changed,prior)
        changed=dict(current,helpers=dict(current['helpers'],**{'build_inputs/rbot_evaluator.cpp':'new generator'}))
        with self.assertRaisesRegex(ValueError,'generator'):model_cache_key(changed,prior)


class InputTests(unittest.TestCase):
    def test_rbot_normalization_preserves_source_and_all_pose_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);poses=root/'poses_first.txt';poses.write_text('header\n'+'1 0 0 0 1 0 0 0 1 0 0 500\n'*1001)
            for body in ['ape','driller']:
                directory=root/body;directory.mkdir();(directory/(body+'.obj')).write_text('v 0 0 0\nv 10 0 0\nv 0 10 0\nf 1 2 3\n')
                images=directory/'frames';images.mkdir()
                for seq in ['a_regular','c_noisy']:
                    for i in range(21):
                        header=b'\x89PNG\r\n\x1a\n'+(13).to_bytes(4,'big')+b'IHDR'+(640).to_bytes(4,'big')+(512).to_bytes(4,'big')+bytes([8,2,0,0,0])+b'\0'*4
                        (images/('%s%04d.png'%(seq,i))).write_bytes(header+b'placeholder')
            before=poses.read_bytes();jobs,identity=data.prepare('rbot',root,root/'cache',True)
            self.assertEqual(len(jobs),4);self.assertEqual(jobs[0]['first_image'],0);self.assertEqual(jobs[0]['input_bits'],8)
            self.assertEqual(jobs[0]['geometry_scale'],.001)
            self.assertEqual(data.base.read_poses(jobs[0]['poses'],1001)[0][-1],.5)
            self.assertEqual(before,poses.read_bytes());self.assertEqual(identity['configurations'],4)

    def test_advanced_scope_and_selected_trajectory_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);root=parent/'交付物/advanced';root.mkdir(parents=True);state=parent/'中间结果文件/run_state_advanced_v1'
            data.write_json(state/'validation/final_audit.json',dict(status='passed',scope='complete_dataset',images=54000))
            data.write_json(state/'independent_audit/inputs.json',dict(status='passed'))
            data.write_json(state/'independent_audit/scenes.json',dict(status='passed',scope='all integer frames',sequences_checked=108,total_frame_checks=54000))
            data.write_json(root/'cameras/camera.json',dict(width=900,height=600,frames_per_sequence=500,K=[[1125,0,449.5],[0,1125,299.5],[0,0,1]],translation_unit='meter',pose_direction='p_camera = R @ p_object + t'))
            rows=[]
            for s,objects in [('scene_01',[1,2,3]),('scene_04',[10,11,12]),('scene_05',[13,14,15])]:
                for t in ['t01','t02','t03']:
                    for c in ['B00','C01','C02','C03','R01','R02','R03','O01','O02','O03','M01','M02']:
                        for n,obj in enumerate(objects):rows.append(dict(scene=s,trajectory=t,condition=c,object='obj_%06d'%obj,is_focus=n==0,
                            sequence_id=s+'_'+t+'_'+c,motion_mode='camera_and_objects_motion',rgb_path='images/'+s+'/'+t+'/'+c))
            data.write_json(state/'independent_audit/evaluation_index.json',dict(configurations=rows))
            selected,audit=data.select('industrial_advanced',root,True)
            self.assertEqual(len(selected),3);self.assertEqual({r['condition'] for r in selected},{'B00','O03','M02'})
            self.assertTrue(all('t01' in r['rgb_path'] for r in selected));self.assertEqual(len(audit),4)
            all_rows,_=data.select('industrial_advanced',root,False);self.assertEqual(len(all_rows),324)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.folder=self.root/'native';pose_dir=self.folder/'results/poses';pose_dir.mkdir(parents=True)
        self.path=pose_dir/'poses_a_regular_ape.csv'
        poses=self.root/'truth.txt';poses.write_text('1 0 0 0 1 0 0 0 1 0 0 0.5\n'*21)
        self.job=dict(config=dict(id='paired',condition='a_regular',object='ape',scene='RBOT',motion='rbot'),object='ape',sequence='a_regular',frames=21,
                      first_image=0,leading_zeros=4,prefix='a_regular',geometry_scale=.001,poses=str(poses),pose_rows=21,intrinsics=data.rbot.INTRINSICS,input_bits=8)
        meta=dict(object='ape',sequence='a_regular',frame_count=20,translation_unit='m',estimate_capture='final_selected_output_before_gt_reset',
                  first_image_index=1,last_image_index=20,model_unit_in_meter=.001,intrinsics=data.rbot.INTRINSICS,
                  translation_threshold_m=.05,rotation_threshold_rad=math.radians(5),input_conversion='native_uint8_bgr')
        data.write_json(pose_dir/'metadata_a_regular_ape.json',meta)
        overlay=self.folder/'results/overlays/initial_gt_image_0.png';overlay.parent.mkdir();overlay.write_bytes(b'placeholder'*20)
        self.variant=data.variants()[0]
        data.write_json(self.folder/'ablation_state.json',dict(variant='A0',flags='00000',prediction_smoothing_independent_of_prior=1,
            cad_matching_enabled=0,cad_renderer_created=0,video_orb=1,video_only_original_accumulation=1,extra_recovery_enabled=0))
        self.rows=[]
        for i in range(20):
            failure=i==2;row=dict(frame=i,image_index=i+1,image_file='a_regular%04d.png'%(i+1),segment_id=int(i>2),estimate_source='single',pose_finite=1,
                tracking_success=int(not failure),reset_primary_after_frame=int(failure),reset_secondary_after_frame=0,translation_error_m=.2 if failure else 0,rotation_error_rad=0)
            for prefix in ['est','gt']:
                row.update({'%s_r%d%d'%(prefix,r,c):int(r==c) for r in range(3) for c in range(3)})
                row.update({prefix+'_t'+a+'_m':v for a,v in zip('xyz',[.2 if failure and prefix=='est' else 0,0,.5])})
            row.update(dict.fromkeys(TIMES,0));row.update(decode_us=9000,preprocessing_us=10,compute_wall_us=1000,tracking_wall_us=10000,frame_wall_us=10020,metric_us=20)
            self.rows.append(row)
        self.write()

    def write(self,folder=None,rows=None):
        folder=folder or self.folder;path=folder/'results/poses/poses_a_regular_ape.csv';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(rows or self.rows)

    def test_failure_counted_before_reset_and_decode_excluded(self):
        result,times=analyze(self.folder,self.job,self.variant)
        self.assertEqual(result['successes'],19);self.assertEqual(result['resets'],1);self.assertEqual(times.shape,(20,len(TIMES)))
        self.assertEqual(times[0,TIMES.index('compute_wall_us')],1000)

    def test_post_reset_pose_cannot_hide_failure(self):
        self.rows[2]['est_tx_m']=0;self.write()
        with self.assertRaisesRegex(ValueError,'pre-reset'):analyze(self.folder,self.job,self.variant)

    def test_runtime_flags_must_match_variant(self):
        other=dict(self.variant,flags='11111')
        with self.assertRaisesRegex(ValueError,'variant'):analyze(self.folder,self.job,other)

    def test_decode_in_compute_is_rejected(self):
        self.rows[0]['compute_wall_us']=10000;self.write()
        with self.assertRaisesRegex(ValueError,'timing'):analyze(self.folder,self.job,self.variant)

    def test_regression_checks_pose_and_segment_but_ignores_timing(self):
        other=self.root/'reference';changed=[dict(r,compute_wall_us=99999) for r in self.rows];self.write(other,changed)
        self.assertTrue(regression(self.folder,other)['passed'])
        changed[0]['est_tx_m']=.001;self.write(other,changed)
        report=regression(self.folder,other)
        self.assertFalse(report['passed'])
        self.assertEqual(report['first_over_tolerance']['row_index'],0)
        self.assertEqual(len(report['first_six_pose_differences']),6)
        changed=[dict(r) for r in self.rows];changed[3]['segment_id']=0;self.write(other,changed)
        self.assertFalse(regression(self.folder,other)['passed'])

    def test_paired_report_preserves_counts_and_drops_only_timing_warmup(self):
        record,times=analyze(self.folder,self.job,self.variant)
        records=[(dict(record,variant=v['id']),times.copy()) for v in data.variants()]
        summary=summarize(records,self.root,'rbot',{},True)
        self.assertEqual(summary['variants']['A1']['success_percent'],95)
        self.assertEqual(summary['variants']['A8']['timing']['compute_wall_us']['frames'],10)
        self.assertEqual(summary['variants']['A8']['timing']['compute_wall_us']['mean_us'],1000)
        self.assertIn('共享负载',(self.root/'results.md').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError,'paired'):summarize(records[:-1],self.root,'rbot',{},True)


if __name__=='__main__':unittest.main()
