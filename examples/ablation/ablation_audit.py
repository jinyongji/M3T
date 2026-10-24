"""Audit saved estimates before GT reset and compare A0/A1 with frozen references."""
import csv
import math
from pathlib import Path
import numpy as np
from ablation_data import base,load,write_json

TIMES=['cycle_us','decode_us','preprocessing_us','tracking_wall_us','compute_wall_us','metric_us','reset_us','frame_wall_us',
       'before_camera_us','correspondences_us','gradient_hessian_us','optimization_us','results_us']


def pose(row,prefix):
    return np.asarray([float(row['%s_r%d%d'%(prefix,r,c)]) for r in range(3) for c in range(3)]+
                      [float(row[prefix+'_t'+a+'_m']) for a in 'xyz'])


def errors(estimate,gt):
    if not np.isfinite(estimate).all():return math.inf,math.inf
    r=estimate[:9].reshape(3,3)
    if np.linalg.det(r)<=1e-8:return math.inf,math.inf
    u,_,v=np.linalg.svd(r);projected=u@v
    if np.linalg.det(projected)<=0:return math.inf,math.inf
    g=gt[:9].reshape(3,3);a,_,b=np.linalg.svd(g);g=a@b
    return float(np.linalg.norm(estimate[9:]-gt[9:])),math.acos(np.clip((np.sum(projected*g)-1)/2,-1,1))


def read_csv(folder):
    files=list((folder/'results/poses').glob('poses_*.csv'))
    if len(files)!=1:raise ValueError('Expected one pose CSV: '+str(folder))
    with files[0].open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    return files[0],rows


def analyze(folder,job,variant=None):
    folder=Path(folder);path,rows=read_csv(folder);meta=load(path.with_name(path.name.replace('poses_','metadata_',1)).with_suffix('.json'))
    expected=dict(object=job['object'],sequence=job['sequence'],frame_count=job['frames']-1,translation_unit='m',
                  estimate_capture='final_selected_output_before_gt_reset',first_image_index=job['first_image']+1,
                  last_image_index=job['first_image']+job['frames']-1)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Unexpected pose metadata: '+str(folder))
    if abs(meta['model_unit_in_meter']-job['geometry_scale'])>1e-8:raise ValueError('Geometry scale mismatch')
    if any(abs(meta['intrinsics'][k]-v)>1e-4 for k,v in job['intrinsics'].items()):raise ValueError('Calibration mismatch')
    if abs(meta['translation_threshold_m']-.05)>1e-7 or abs(meta['rotation_threshold_rad']-math.radians(5))>1e-7:raise ValueError('Threshold mismatch')
    if len(rows)!=job['frames']-1:raise ValueError('Incomplete pose rows')
    if variant:
        state=load(folder/'ablation_state.json')
        for key,value in dict(variant=variant['id'],flags=variant['flags'],prediction_smoothing_independent_of_prior=1,
                              cad_matching_enabled=int(variant['flags'][0]),cad_renderer_created=int(variant['flags'][0]),video_orb=1,
                              video_only_original_accumulation=int(variant['flags'][0]=='0'),extra_recovery_enabled=0).items():
            if state.get(key)!=value:raise ValueError('Runtime variant does not match launch: '+key)
        if meta.get('input_conversion')!=('native_uint8_bgr' if job['input_bits']==8 else 'uint16_to_uint8_fixed_1_over_257'):
            raise ValueError('Native input conversion mismatch')
    truth=np.asarray(base.read_poses(job['poses'],job['pose_rows']))
    segment=passed=again=invalid=0;differences=[];timing=[]
    for i,row in enumerate(rows):
        image=job['first_image']+i+1;filename=job['prefix']+str(image).zfill(job['leading_zeros'])+'.png'
        if int(row['frame'])!=i or int(row['image_index'])!=image or row['image_file']!=filename or int(row['segment_id'])!=segment or row['estimate_source']!='single':
            raise ValueError('Image/segment alignment mismatch')
        for name in ['pose_finite','tracking_success','reset_primary_after_frame','reset_secondary_after_frame']:
            if row[name] not in ['0','1']:raise ValueError('Invalid flag')
        success=int(row['tracking_success']);reset=int(row['reset_primary_after_frame'])
        if reset!=1-success or int(row['reset_secondary_after_frame'])!=0:raise ValueError('Invalid benchmark reset')
        gt=pose(row,'gt');est=pose(row,'est')
        if not np.isfinite(gt).all() or not np.allclose(gt,truth[i+1],atol=2e-6,rtol=0):raise ValueError('CSV GT differs from source')
        if int(row['pose_finite'])!=int(np.isfinite(est).all()):raise ValueError('Invalid finite flag')
        te,re=errors(est,gt);valid=math.isfinite(te) and math.isfinite(re)
        if not valid:invalid+=1
        audited=int(valid and te<=.05 and re<=math.radians(5))
        nt,nr=float(row['translation_error_m']),float(row['rotation_error_rad'])
        if valid:
            if not math.isfinite(nt) or not math.isfinite(nr) or abs(nt-te)>3e-6 or abs(nr-re)>.003:
                raise ValueError('Native errors do not match saved pre-reset estimate')
        elif success:raise ValueError('Invalid estimate counted successful')
        if success!=audited:differences.append(image)
        if success and (nt>.05+1e-8 or nr>math.radians(5)+1e-7):raise ValueError('Native success contradicts native errors')
        t={k:float(row[k]) for k in TIMES}
        if any(not math.isfinite(x) or x<0 for x in t.values()):raise ValueError('Invalid timing')
        close=lambda a,b:abs(a-b)<=max(2,abs(a)*1e-4)
        if not close(t['compute_wall_us'],t['tracking_wall_us']-t['decode_us']) or not close(t['frame_wall_us'],t['tracking_wall_us']+t['metric_us']+t['reset_us']) or t['preprocessing_us']>t['compute_wall_us']+2:
            raise ValueError('Compute/decode timing mismatch')
        passed+=success;again+=audited;segment+=reset;timing.append(t)
    overlay=folder/'results/overlays'/('initial_gt_image_%d.png'%job['first_image'])
    if not overlay.is_file() or overlay.stat().st_size<100:raise ValueError('Missing initial GT overlay')
    result=dict(job['config'],frames=len(rows),successes=passed,recomputed_successes=again,resets=segment,invalid=invalid,
                audit_disagreements=differences,pose_csv=str(path),initial_overlay=str(overlay))
    write_json(folder/'audit.json',result)
    return result,np.asarray([[t[k] for k in TIMES] for t in timing],dtype=float)


def regression(new_folder,old_folder):
    _,new=read_csv(Path(new_folder));_,old=read_csv(Path(old_folder))
    if len(new)!=len(old):raise ValueError('Regression row mismatch')
    different=[];segments=[];worst_r=worst_t=0.;first_six=[];first_over=None
    for i,(a,b) in enumerate(zip(new,old)):
        if a['image_file']!=b['image_file']:raise ValueError('Regression image mismatch')
        ea,eb=pose(a,'est'),pose(b,'est')
        if not np.isfinite(ea).all() or not np.isfinite(eb).all():different.append(i);continue
        rotation=float(np.max(np.abs(ea[:9]-eb[:9])));translation=float(np.linalg.norm(ea[9:]-eb[9:]))
        delta=dict(row_index=i,image_file=a['image_file'],maximum_rotation_matrix_entry_difference=rotation,translation_difference_m=translation)
        if i<6:first_six.append(delta)
        if first_over is None and (rotation>1e-4 or translation>5e-5):first_over=delta
        worst_r=max(worst_r,rotation);worst_t=max(worst_t,translation)
        if a['tracking_success']!=b['tracking_success'] or a['reset_primary_after_frame']!=b['reset_primary_after_frame']:different.append(i)
        if a['segment_id']!=b['segment_id']:segments.append(i)
    passed=not different and not segments and worst_r<=1e-4 and worst_t<=5e-5
    return dict(passed=passed,frames=len(new),classification_or_invalid_rows=different,segment_rows=segments,
                maximum_rotation_matrix_entry_difference=worst_r,maximum_translation_difference_m=worst_t,
                first_six_pose_differences=first_six,first_over_tolerance=first_over,row_index_base=0,
                tolerance_rotation_entry=1e-4,tolerance_translation_m=5e-5,
                note='Representative smoke regression only; no comparison of shared-host wall times')
