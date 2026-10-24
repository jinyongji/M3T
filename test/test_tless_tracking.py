"""Geometry, symmetry, instance ordering and pre-reset audit invariants."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/tless'))
import tracking_data as data
from tracking_report import analyze,summarize,TIMES


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.points=np.array([[0,0,0],[.02,0,0],[0,.02,0],[-.02,0,0],[0,-.02,0]])
        self.k=np.array([[1000.,0,300],[0,1000.,250],[0,0,1]])
        self.gt=np.eye(4);self.gt[2,3]=1

    def test_ascii_conversion_preserves_origin_scale_and_faces(self):
        p=self.root/'mesh.ply'
        p.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n10 20 30\n20 20 30\n10 30 30\n3 0 1 2\n')
        xyz,tri=data.ply(p,True)
        np.testing.assert_allclose(xyz,[[.01,.02,.03],[.02,.02,.03],[.01,.03,.03]])
        self.assertEqual(tri,[[0,1,2]])

    def test_binary_evaluation_vertices_and_truncation(self):
        p=self.root/'mesh.ply';header=b'ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
        p.write_bytes(header+struct.pack('<6f',1,2,3,4,5,6))
        np.testing.assert_allclose(data.ply(p)[0],np.array([[1,2,3],[4,5,6]])*.001)
        p.write_bytes(header+struct.pack('<3f',1,2,3))
        with self.assertRaisesRegex(ValueError,'Truncated'):data.ply(p)

    def test_cad_normals_are_preserved_without_millimetre_scaling(self):
        p=self.root/'normal.ply'
        p.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n10 20 30 0 0 1\n20 20 30 0 0 1\n10 30 30 0 0 1\n3 0 1 2\n')
        _,_,normals=data.ply(p,True,True)
        np.testing.assert_array_equal(normals,[[0,0,1]]*3)

    def test_metric_is_maximum_not_average_distance(self):
        p=self.gt.copy();p[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        mssd,mspd=data.metric(p,self.gt,self.points,np.eye(4)[None],self.k)
        self.assertAlmostEqual(mssd,math.sqrt(2)*.02,12);self.assertAlmostEqual(mspd,math.sqrt(2)*20,10)

    def test_translation_has_expected_metric_and_pixel_units(self):
        p=self.gt.copy();p[0,3]=.003
        mssd,mspd=data.metric(p,self.gt,self.points,np.eye(4)[None],self.k)
        self.assertAlmostEqual(mssd,.003,12);self.assertAlmostEqual(mspd,3,10)

    def test_off_origin_discrete_symmetry_keeps_translation(self):
        s=np.eye(4);s[:3,:3]=np.diag([-1,-1,1]);s[0,3]=20
        syms=data.symmetries(dict(symmetries_discrete=[s.ravel().tolist()]))
        self.assertEqual(len(syms),2);self.assertAlmostEqual(syms[1,0,3],.02)
        self.assertLess(max(data.metric(self.gt@syms[1],self.gt,self.points,syms,self.k)),1e-9)

    def test_continuous_axis_offset_and_discretization(self):
        syms=data.symmetries(dict(symmetries_continuous=[dict(axis=[0,0,1],offset=[10,0,0])]))
        self.assertEqual(len(syms),math.ceil(math.pi/.01))
        self.assertGreater(np.abs(syms[:,:3,3]).max(),.015)
        self.assertLess(max(data.metric(self.gt@syms[20],self.gt,self.points,syms,self.k)),1e-9)

    def test_pruned_metric_equals_exhaustive_symmetry_formula(self):
        rng=np.random.default_rng(7);points=rng.normal(size=(50,3))*.02
        syms=data.symmetries(dict(symmetries_continuous=[dict(axis=[0,0,1],offset=[10,0,0])]),.2)
        p=self.gt.copy();a=.31;p[:3,:3]=[[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]];p[0,3]=.004
        got=data.metric(p,self.gt,points,syms,self.k)
        est=points@p[:3,:3].T+p[:3,3];uv=est[:,:2]/est[:,2,None]*1000+[300,250]
        scores=[]
        for s in syms:
            target=self.gt@s;g=points@target[:3,:3].T+target[:3,3]
            pixels=g[:,:2]/g[:,2,None]*1000+[300,250]
            scores.append([np.linalg.norm(est-g,axis=1).max(),np.linalg.norm(uv-pixels,axis=1).max()])
        np.testing.assert_allclose(got,np.min(scores,axis=0),atol=1e-10)

    def test_invalid_reflection_or_nonfinite_is_failure(self):
        for p in [self.gt*np.nan,np.diag([-1.,1,1,1])]:
            self.assertTrue(all(math.isinf(x) for x in data.metric(p,self.gt,self.points,np.eye(4)[None],self.k)))

    def test_lossless_integer_padding_preserves_projection_differences(self):
        original=self.k.copy();canonical=original.copy();canonical[:2,2]+=[37,95]
        point=np.array([.1,-.2,1]);u=original@point;v=canonical@point
        np.testing.assert_array_equal(v[:2]-u[:2],[37,95])
        picture=np.arange(3*4*3).reshape(3,4,3);canvas=np.full((8,9,3),127);canvas[2:5,3:7]=picture
        np.testing.assert_array_equal(canvas[2:5,3:7],picture)


def synthetic_scene():
    cameras={};gt={};info={}
    for ring,elev in enumerate(data.PROTOCOL['elevations']):
        for j in range(72):
            i=ring*72+j;a=2*math.pi*j/72;r=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
            t=np.array([.5,0,1]);cameras[str(i)]=dict(cam_R_w2c=r.ravel().tolist(),cam_t_w2c=(t*1000).tolist(),elev=elev,mode=0)
            gt[str(i)]=[dict(obj_id=1,cam_R_m2c=r.ravel().tolist(),cam_t_m2c=((r@np.array([offset,0,0])+t)*1000).tolist()) for offset in [0,.05]]
            info[str(i)]=[dict(visib_fract=0 if j<3 else .8),dict(visib_fract=.6)]
    return {1:(cameras,gt,info)}


class SequenceTests(unittest.TestCase):
    def test_initialization_first_observable_then_cyclic_complete_ring(self):
        configs,audit=data.make_index(synthetic_scene(),np.eye(3))
        self.assertEqual(len(configs),14);self.assertEqual(configs[0]['image_ids'][0],3)
        self.assertEqual(set(configs[0]['image_ids']),set(range(72)));self.assertLess(audit['world_translation_max_m'],1e-12)

    def test_same_class_index_permutation_rejected_by_world_pose(self):
        scenes=synthetic_scene();scenes[1][1]['5'].reverse()
        with self.assertRaisesRegex(ValueError,'instance/world-pose'):data.make_index(scenes,np.eye(3))

    def test_wrong_view_order_is_not_silently_stitched(self):
        scenes=synthetic_scene();cams=scenes[1][0]
        # Apply a camera and matching object/GT permutation, preserving world invariance.
        for mapping in scenes[1]:mapping['1'],mapping['20']=mapping['20'],mapping['1']
        with self.assertRaisesRegex(ValueError,'gap/order'):data.make_index(scenes,np.eye(3))


class OutputAuditTests(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        prepared=root/'prepared';model=prepared/'models/obj_000001';model.mkdir(parents=True)
        points=np.array([[0,0,0],[.02,0,0],[0,.02,0]]);np.savetxt(str(model/'evaluation.xyz'),points)
        np.savetxt(str(model/'symmetries.txt'),np.r_[np.eye(3).ravel(),[0,0,0]][None])
        scene=root/'test_primesense/000001';scene.mkdir(parents=True)
        p=np.eye(4);p[2,3]=1;truth={str(i):[dict(obj_id=1,cam_R_m2c=np.eye(3).ravel().tolist(),cam_t_m2c=[0,0,1000])] for i in range(3)}
        data.save(scene/'scene_gt.json',truth);data.save(scene/'scene_gt_info.json',{str(i):[dict(visib_fract=.8)] for i in range(3)})
        index=dict(models={'1':dict(diameter_m=.04,symmetries=1)},intrinsics=[[1000,0,300],[0,1000,250],[0,0,1]],image_size=[757,635])
        config=dict(id='test',obj_id=1,scene_id=1,elevation=75,gt_index=0,image_ids=[0,1,2])
        folder=root/'rbgt';folder.mkdir();data.save(folder/'metadata.json',dict(method='rbgt',config_id='test',frame_count=2,initial_image_id=0,estimate_capture='before_gt_reset',mssd_reset_fraction=.1,cad_texture_available=0,cad_disk_executed=0,diameter_m=.04,symmetry_count=1,cad_disk_configured=0,intrinsics=[1000,1000,300,250,757,635]))
        rows=[]
        for i,dx in enumerate([.01,0]):
            est=p.copy();est[0,3]=dx
            row=dict(frame=i,image_id=i+1,visibility=.8,segment_id=i,pose_finite=1,success=int(dx==0),reset_after=int(dx!=0),mssd_m=dx,mspd_px=dx*1000)
            for prefix,pose in [('est',est),('gt',p)]:
                row.update({prefix+'_r%d%d'%(r,c):pose[r,c] for r in range(3) for c in range(3)})
                row.update({prefix+'_t'+a:pose[r,3] for r,a in enumerate('xyz')})
            row.update(dict.fromkeys(TIMES,0));row.update(decode_us=10,preprocessing_us=2,tracking_wall_us=15,compute_wall_us=5,metric_us=4,reset_us=1,frame_wall_us=20)
            rows.append(row)
        return root,prepared,index,config,folder,rows

    def write(self,folder,rows):
        with (folder/'poses.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def test_pre_reset_failure_and_segments_audited(self):
        root,p,index,c,folder,rows=self.fixture();self.write(folder,rows)
        result,_,_=analyze(folder,c,index,p,root,3,{},{});self.assertEqual(result['resets'],1);self.assertEqual(result['successes'],1)

    def test_saved_post_reset_truth_cannot_hide_failed_estimate(self):
        root,p,index,c,folder,rows=self.fixture();rows[0]['est_tx']=0;self.write(folder,rows)
        with self.assertRaisesRegex(ValueError,'errors disagree'):analyze(folder,c,index,p,root,3,{},{})

    def test_decode_cannot_be_counted_as_compute(self):
        root,p,index,c,folder,rows=self.fixture();rows[0]['compute_wall_us']=15;self.write(folder,rows)
        with self.assertRaisesRegex(ValueError,'Timing decomposition'):analyze(folder,c,index,p,root,3,{},{})


class SummaryTests(unittest.TestCase):
    def fixture(self):
        records=[]
        for method in data.METHODS:
            errors=[(.05,3,.8,1,False)]*18+[(.2,20,.05,1,True),(math.inf,math.inf,.2,0,False)]
            times=[]
            for i in range(20):
                t=dict.fromkeys(TIMES,0);t['compute_wall_us']=100 if i<10 else 1000
                t['decode_us']=9000;t['tracking_wall_us']=t['compute_wall_us']+9000;times.append(t)
            result=dict(config_id='paired',method=method,frames=20,successes=18,resets=2,recomputed_successes=18,
                        audit_disagreements=[],invalid=1)
            records.append((result,errors,times))
        return records,dict(protocol=data.PROTOCOL,fingerprint='synthetic')

    def test_paired_report_keeps_invalid_and_low_visibility_and_excludes_decode(self):
        records,index=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);summary=summarize(records,index,root,True)
            self.assertEqual(summary['methods']['rbgt']['success_percent'],90)
            item=summary['methods']['ours']
            self.assertEqual(item['timing']['compute_wall_us']['frames'],10)
            self.assertEqual(item['timing']['compute_wall_us']['mean'],1000)
            self.assertEqual(item['timing']['decode_us']['mean'],9000)
            self.assertEqual(item['visibility']['under_10_percent']['frames'],1)
            self.assertEqual(item['native_boundary']['within_8px']['frames'],1)
            self.assertEqual(item['invalid'],1);self.assertIsNone(item['mssd_fraction']['mean'])
            self.assertEqual(data.load(root/'summary.json')['status'],'complete')
            text=(root/'results.md').read_text(encoding='utf-8')
            self.assertIn('ICG+',text);self.assertIn('Ours (CAD-DISK inactive)',text)
            self.assertIn('90.0000%',text)

    def test_missing_or_duplicate_method_cannot_look_paired(self):
        records,index=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            for broken in [records[:-1],records[:-1]+[records[0]]]:
                with self.assertRaisesRegex(ValueError,'Unpaired'):summarize(broken,index,Path(temp),True)


if __name__=='__main__':unittest.main()
