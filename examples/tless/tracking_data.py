"""Frozen T-LESS view-ring protocol, lossless camera normalization and geometry."""
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np

METHODS = ['rbgt','srt3d','icg_rgb','m3t_original_rgb','ours']
LABELS = dict(zip(METHODS,['RBGT','SRT3D','ICG','ICG+','Ours (CAD-DISK inactive)']))
PROTOCOL = dict(version=1, task='GT-initialized static viewpoint-ring tracking; not BOP localization',
                elevations=[75,65,55,45,35,25,15], frames_per_ring=72,
                initialization='first original-id view with visib_fract >= 0.1; cyclic ring order',
                evaluation='all remaining 71 views including low visibility; no inter-ring transitions',
                reset='capture estimate, evaluate MSSD/d <= 0.1, count failure, then reset to that GT',
                symmetry_discretization=0.01, warmup=10, padding_bgr=[127,127,127],
                geometry_conversion='CAD millimetres to OBJ metres; preserve origin, triangle order and vertex normals',
                primary_timing='compute_wall_us; includes actual online integer padding and noise filtering',
                maximum_validated_within_ring_step_deg=10,
                native_boundary_analysis_band_px=8,
                depth_input=False, ground_truth_occluder_input=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rigid(rotation,translation,scale=0.001):
    r=np.asarray(rotation,dtype=float).reshape(3,3);t=np.asarray(translation,dtype=float)
    if t.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(t).all():raise ValueError('Invalid pose')
    if np.max(np.abs(r.T@r-np.eye(3)))>1e-3 or abs(np.linalg.det(r)-1)>1e-3:raise ValueError('Invalid rotation')
    p=np.eye(4);p[:3,:3]=r;p[:3,3]=t*scale;return p


def ply(path,faces=False,normals=False):
    """Read the supplied ASCII CAD or binary little-endian evaluation vertices."""
    with Path(path).open('rb') as f:
        if f.readline().strip()!=b'ply':raise ValueError('Not PLY: '+str(path))
        fmt=None;count=0;face_count=0;properties=[];element=None
        for _ in range(1000):
            line=f.readline().decode('ascii').strip();parts=line.split()
            if not parts:raise ValueError('Malformed PLY header')
            if parts[0]=='format':fmt=parts[1]
            elif parts[0]=='element':
                element=parts[1]
                if element=='vertex':count=int(parts[2])
                elif element=='face':face_count=int(parts[2])
            elif parts[0]=='property' and element=='vertex':properties.append(parts[1:])
            elif line=='end_header':break
        else:raise ValueError('Missing PLY end_header')
        if count<=0 or count>2000000 or any(len(p)!=2 for p in properties):raise ValueError('Unsupported vertex schema')
        names=[p[1] for p in properties]
        if not all(n in names for n in ['x','y','z']):raise ValueError('Missing XYZ')
        if fmt=='ascii':
            values=np.asarray([[float(v) for v in f.readline().split()] for _ in range(count)],dtype=float)
            if values.shape!=(count,len(properties)):raise ValueError('Malformed ASCII vertices')
        elif fmt=='binary_little_endian':
            types={'float':'<f4','float32':'<f4','double':'<f8','uchar':'u1','uint8':'u1'}
            dtype=np.dtype([(n,types[t]) for t,n in properties])
            raw=f.read(count*dtype.itemsize)
            if len(raw)!=count*dtype.itemsize:raise ValueError('Truncated binary vertices')
            arr=np.frombuffer(raw,dtype=dtype);values=np.column_stack([arr[n] for n in names])
        else:raise ValueError('Unsupported PLY format: '+str(fmt))
        points=values[:,[names.index(n) for n in ['x','y','z']]]*.001
        if not np.isfinite(points).all():raise ValueError('Nonfinite vertices')
        triangles=[]
        if faces:
            if fmt!='ascii':raise ValueError('CAD converter requires supplied ASCII triangles')
            for _ in range(face_count):
                row=[int(v) for v in f.readline().split()]
                if len(row)!=4 or row[0]!=3 or min(row[1:])<0 or max(row[1:])>=count:raise ValueError('Invalid/nontriangular CAD face')
                triangles.append(row[1:])
            if not triangles:raise ValueError('No CAD triangles')
        if normals:
            if not all(n in names for n in ['nx','ny','nz']):raise ValueError('CAD normals missing')
            n=values[:,[names.index(n) for n in ['nx','ny','nz']]]
            if not np.isfinite(n).all():raise ValueError('Invalid normals')
            return points,triangles,n
        return points,triangles


def symmetries(info,step=0.01):
    """BOP-style finite SE(3) symmetry set; retain off-origin translation offsets."""
    discrete=[np.eye(4)]
    for value in info.get('symmetries_discrete',[]):
        p=np.asarray(value,dtype=float).reshape(4,4).copy();p[:3,3]*=.001;discrete.append(p)
    axes=info.get('symmetries_continuous',[])
    if len(axes)>1:raise ValueError('Protocol supports the supplied single continuous axis')
    continuous=[]
    for entry in axes:
        axis=np.asarray(entry['axis'],dtype=float);axis/=np.linalg.norm(axis)
        offset=np.asarray(entry['offset'],dtype=float)*.001
        skew=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        for angle in np.arange(math.ceil(math.pi/step))*2*math.pi/math.ceil(math.pi/step):
            r=np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)
            p=np.eye(4);p[:3,:3]=r;p[:3,3]=offset-r@offset;continuous.append(p)
    return np.asarray([c@d for d in discrete for c in (continuous or [np.eye(4)])])


def metadata(root,expected=None):
    root=Path(root);sources={}
    for directory in ['models_cad','models_eval']:
        for name in ['models_info.json']+['obj_%06d.ply'%i for i in range(1,31)]:
            p=root/directory/name;sources[p.relative_to(root).as_posix()]=sha(p)
    scenes={};all_k=[]
    for scene_id in range(1,21):
        folder=root/'test_primesense'/('%06d'%scene_id)
        parts=[]
        for name in ['scene_camera.json','scene_gt.json','scene_gt_info.json']:
            p=folder/name;sources[p.relative_to(root).as_posix()]=sha(p);parts.append(load(p))
        cameras,truth,visibility=parts
        if set(cameras)!=set(truth) or set(truth)!=set(visibility) or set(map(int,cameras))!=set(range(504)):
            raise ValueError('Incomplete scene metadata')
        scenes[scene_id]=parts
        all_k.extend(np.asarray(c['cam_K'],dtype=float).reshape(3,3) for c in cameras.values())
    if expected is not None and sources!=expected:raise ValueError('Dataset metadata/models differ from the inspected server export')
    ks=np.asarray(all_k);common=ks[0].copy()
    if (not np.isfinite(ks).all() or np.max(np.abs(ks[:,0,0]-common[0,0]))>1e-8
            or np.max(np.abs(ks[:,1,1]-common[1,1]))>1e-8
            or np.max(np.abs(ks[:,2,:]-[0,0,1]))>1e-8 or np.max(np.abs(ks[:,0,1]))>1e-8
            or np.max(np.abs(ks[:,1,0]))>1e-8):raise ValueError('This lossless normalization requires fixed focal lengths and zero skew')
    principal=ks[:,[0,1],[2,2]];maximum=principal.max(0);shifts=maximum-principal
    if np.max(np.abs(shifts-np.rint(shifts)))>1e-7:raise ValueError('Principal point shifts are not integer pixels')
    common[:2,2]=maximum
    dimensions=[720+int(round(shifts[:,0].max())),540+int(round(shifts[:,1].max()))]
    return sources,scenes,common,dimensions


def make_index(scenes,k):
    configs=[];max_translation=0.;max_rotation=0.;max_step=0.;max_azimuth=0.;counts={}
    for scene_id,(cameras,truth,visibility) in sorted(scenes.items()):
        anchors=[]
        first=cameras['0'];world_camera=rigid(first['cam_R_w2c'],first['cam_t_w2c'])
        for p in truth['0']:anchors.append(np.linalg.inv(world_camera)@rigid(p['cam_R_m2c'],p['cam_t_m2c']))
        ids0=[p['obj_id'] for p in truth['0']]
        counts[scene_id]=len(ids0)
        for image_id in range(504):
            cam=cameras[str(image_id)];wc=rigid(cam['cam_R_w2c'],cam['cam_t_w2c'])
            if [p['obj_id'] for p in truth[str(image_id)]]!=ids0:raise ValueError('Instance ordering changed; explicit assignment required')
            if len(visibility[str(image_id)])!=len(ids0):raise ValueError('GT visibility index mismatch')
            for j,p in enumerate(truth[str(image_id)]):
                world=np.linalg.inv(wc)@rigid(p['cam_R_m2c'],p['cam_t_m2c'])
                dt=np.linalg.norm(world[:3,3]-anchors[j][:3,3]);dr=np.max(np.abs(world[:3,:3]-anchors[j][:3,:3]))
                max_translation=max(max_translation,float(dt));max_rotation=max(max_rotation,float(dr))
                if dt>1e-5 or dr>2e-5:raise ValueError('Physical instance/world-pose invariant failed; do not trust list index')
        for elev in PROTOCOL['elevations']:
            ids=sorted(int(i) for i,c in cameras.items() if c.get('elev')==elev and c.get('mode')==0)
            if len(ids)!=72:raise ValueError('Expected seven separate 72-view rings')
            centers=[]
            for i in ids:
                p=rigid(cameras[str(i)]['cam_R_w2c'],cameras[str(i)]['cam_t_w2c']);centers.append(-p[:3,:3].T@p[:3,3])
            azimuth=np.unwrap(np.arctan2(np.asarray(centers)[:,1],np.asarray(centers)[:,0]))
            increments=np.diff(azimuth);direction=1 if np.median(increments)>0 else -1
            circular=np.r_[increments, direction*2*np.pi-(azimuth[-1]-azimuth[0])]*direction
            max_azimuth=max(max_azimuth,float(np.rad2deg(circular.max())))
            if circular.min()<np.deg2rad(.5) or circular.max()>np.deg2rad(10):raise ValueError('View ring has an unvalidated angular gap/order')
            for a,b in zip(ids,ids[1:]+ids[:1]):
                ra=np.asarray(cameras[str(a)]['cam_R_w2c']).reshape(3,3);rb=np.asarray(cameras[str(b)]['cam_R_w2c']).reshape(3,3)
                step=math.degrees(math.acos(np.clip((np.trace(ra.T@rb)-1)/2,-1,1)))
                max_step=max(max_step,step)
                if step>PROTOCOL['maximum_validated_within_ring_step_deg']:raise ValueError('Orientation jump inside ring')
            for j,obj_id in enumerate(ids0):
                available=[q for q,i in enumerate(ids) if visibility[str(i)][j]['visib_fract']>=.1]
                if not available:raise ValueError('No observable initialization; record an explicit protocol exclusion')
                start=available[0];order=ids[start:]+ids[:start]
                configs.append(dict(id='scene_%02d__elev_%02d__instance_%02d__obj_%06d'%(scene_id,elev,j,obj_id),
                                    scene_id=scene_id,elevation=elev,gt_index=j,obj_id=obj_id,image_ids=order))
    return configs,dict(instance_counts=counts,world_translation_max_m=max_translation,
                        world_rotation_max_entry_difference=max_rotation,adjacent_rotation_max_deg=max_step,
                        adjacent_center_azimuth_max_deg=max_azimuth,
                        instances=sum(counts.values()),rings=140,configurations=len(configs))


def prepare(root,destination,expected=None):
    root=Path(root);destination=Path(destination);sources,scenes,k,wh=metadata(root,expected)
    configs,audit=make_index(scenes,k)
    signature=hashlib.sha256(json.dumps(dict(sources=sources,protocol=PROTOCOL),sort_keys=True).encode()).hexdigest()
    destination=destination/signature[:16];destination.mkdir(parents=True,exist_ok=True)
    info=load(root/'models_eval/models_info.json');models={}
    for obj_id in range(1,31):
        name='obj_%06d'%obj_id;points,faces,normals=ply(root/'models_cad'/(name+'.ply'),True,True)
        model=destination/'models'/name;model.mkdir(parents=True,exist_ok=True)
        with (model/'model.obj').open('w',encoding='ascii',newline='\n') as f:
            for v in points:f.write('v %.12g %.12g %.12g\n'%tuple(v))
            for v in normals:f.write('vn %.12g %.12g %.12g\n'%tuple(v))
            for face in faces:f.write('f '+' '.join('%d//%d'%(i+1,i+1) for i in face)+'\n')
        evaluation,_=ply(root/'models_eval'/(name+'.ply'))
        # Reorder vertices only, preserving the exact full maximum-distance set.
        axis=info[str(obj_id)].get('symmetries_continuous',[])
        if axis:
            a=np.asarray(axis[0]['axis']);o=np.asarray(axis[0]['offset'])*.001;v=evaluation-o
            priority=np.linalg.norm(v-np.outer(v@a,a),axis=1)
        else:priority=np.linalg.norm(evaluation,axis=1)
        evaluation=evaluation[np.argsort(-priority,kind='stable')]
        np.savetxt(str(model/'evaluation.xyz'),evaluation,fmt='%.17g')
        syms=symmetries(info[str(obj_id)])
        with (model/'symmetries.txt').open('w',encoding='ascii',newline='\n') as f:
            for s in syms:f.write(' '.join('%.17g'%v for v in np.r_[s[:3,:3].ravel(),s[:3,3]])+'\n')
        models[str(obj_id)]=dict(diameter_m=info[str(obj_id)]['diameter']*.001,vertices=len(points),evaluation_vertices=len(evaluation),
                            symmetries=len(syms),has_texture=False,model_sha256=sha(model/'model.obj'),
                            evaluation_sha256=sha(model/'evaluation.xyz'),symmetries_sha256=sha(model/'symmetries.txt'))
    (destination/'camera.txt').write_text(' '.join('%.17g'%v for v in [k[0,0],k[1,1],k[0,2],k[1,2],*wh])+'\n',encoding='ascii')
    for c in configs:
        cams,gt,vis=scenes[c['scene_id']];folder=destination/'configurations'/c['id'];folder.mkdir(parents=True,exist_ok=True)
        lines=['obj_%06d %.17g %d'%(c['obj_id'],models[str(c['obj_id'])]['diameter_m'],len(c['image_ids']))]
        for i in c['image_ids']:
            key=str(i);p=gt[key][c['gt_index']];pose=rigid(p['cam_R_m2c'],p['cam_t_m2c']);cam=np.asarray(cams[key]['cam_K']).reshape(3,3)
            shift=np.rint(k[:2,2]-cam[:2,2]).astype(int)
            lines.append('%d %d %d %.17g '%(i,shift[0],shift[1],vis[key][c['gt_index']]['visib_fract'])+
                         ' '.join('%.17g'%v for v in np.r_[pose[:3,:3].ravel(),pose[:3,3]]))
        path=folder/'frames.txt';path.write_text('\n'.join(lines)+'\n',encoding='ascii');c['frames_sha256']=sha(path)
    manifest=dict(status='prepared_metadata',fingerprint=signature,source_sha256=sources,protocol=PROTOCOL,
                  configurations=configs,models=models,intrinsics=k.tolist(),image_size=wh,metadata_audit=audit,
                  evaluated_frames_per_method=sum(len(c['image_ids'])-1 for c in configs),
                  timed_frames_per_method=sum(len(c['image_ids'])-1-PROTOCOL['warmup'] for c in configs),
                  camera_sha256=sha(destination/'camera.txt'))
    save(destination/'index.json',manifest)
    return destination,manifest


def rgb_preflight(root,configs,frames):
    records={}
    for c in configs:
        for image_id in c['image_ids'][:frames]:
            name='test_primesense/%06d/rgb/%06d.png'%(c['scene_id'],image_id)
            if name in records:continue
            p=Path(root)/name
            with p.open('rb') as f:b=f.read(33)
            if (len(b)!=33 or b[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',b[16:24])!=(720,540)
                    or b[24]!=8 or b[25]!=2):raise ValueError('Expected native 720x540 8-bit RGB PNG: '+name)
            s=p.stat();records[name]=[s.st_size,s.st_mtime_ns,hashlib.sha256(b).hexdigest()]
    return dict(files=len(records),fingerprint=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
                scope='all selected PNG headers, sizes and mtimes; metadata/meshes have full SHA256')


def metric(pose,truth,vertices,syms,k):
    """Independent NumPy MSSD/MSPD, exact full vertices and discretized symmetries."""
    if not np.isfinite(pose).all() or np.linalg.det(pose[:3,:3])<=1e-8:return math.inf,math.inf
    u,_,v=np.linalg.svd(pose[:3,:3]);r=u@v
    if np.linalg.det(r)<=0:return math.inf,math.inf
    estimated=vertices@r.T+pose[:3,3]
    if (estimated[:,2]<=0).any():return math.inf,math.inf
    uv=estimated[:,:2]/estimated[:,2,None]*[k[0,0],k[1,1]]+k[:2,2]
    probe=np.unique(np.linspace(0,len(vertices)-1,min(24,len(vertices))).astype(int))
    gt_sym=np.einsum('ij,sjk->sik',truth,syms)
    q=np.einsum('vj,sij->svi',vertices[probe],gt_sym[:,:3,:3])+gt_sym[:,:3,3,None].transpose(0,2,1)
    if (q[:,:,2]<=0).any():return math.inf,math.inf
    d3=np.linalg.norm(q-estimated[probe],axis=2).max(1)
    up=q[:,:,:2]/q[:,:,2,None]*[k[0,0],k[1,1]]+k[:2,2]
    d2=np.linalg.norm(up-uv[probe],axis=2).max(1)
    best3=best2=math.inf
    for j in np.argsort(d3):
        if d3[j]>best3 and d2[j]>best2:continue
        points=vertices@gt_sym[j,:3,:3].T+gt_sym[j,:3,3]
        best3=min(best3,float(np.linalg.norm(points-estimated,axis=1).max()))
        pixels=points[:,:2]/points[:,2,None]*[k[0,0],k[1,1]]+k[:2,2]
        best2=min(best2,float(np.linalg.norm(pixels-uv,axis=1).max()))
    return best3,best2
