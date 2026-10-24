"""Publish completed moving-camera/moving-object sequences with backup and rollback."""
from pathlib import Path,PurePosixPath
from contextlib import contextmanager
import os,json,shutil,hashlib,time

MODE='camera_and_objects_motion'
REVISION='v0.9-20261017'
OLD_NAME='工业物体跟踪数据集-v0.8-20261015'

def write_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');os.replace(tmp,path)

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def inside(path,root):
 path=Path(path).resolve();root=Path(root).resolve()
 if path==root or root not in path.parents:raise RuntimeError(f'Path is outside the permitted dataset directory: {path}')
 return path

@contextmanager
def exclusive_lock(path):
 path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+b')
 try:
  if os.name=='nt':
   import msvcrt
   f.seek(0)
   if not f.read(1):f.write(b'1');f.flush()
   f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  yield
 finally:f.close()

def link_or_copy(source,destination):
 source=Path(source);destination=Path(destination)
 if source.is_symlink():raise RuntimeError(f'Symlink input is not supported: {source}')
 destination.parent.mkdir(parents=True,exist_ok=True)
 try:os.link(source,destination)
 except OSError:shutil.copy2(source,destination)

def prepare_copy(source,destination):
 if source.is_dir():shutil.copytree(source,destination,copy_function=link_or_copy)
 else:link_or_copy(source,destination)

def same_content(a,b):
 if a.is_dir():
  if not b.is_dir():return False
  aa={p.relative_to(a) for p in a.rglob('*') if p.is_file()};bb={p.relative_to(b) for p in b.rglob('*') if p.is_file()}
  return aa==bb and all(same_content(a/p,b/p) for p in aa)
 if not b.is_file() or a.stat().st_size!=b.stat().st_size:return False
 return os.path.samefile(a,b) or sha(a)==sha(b)

def rollback(journal,root,work):
 data=json.loads(journal.read_text(encoding='utf8'))
 if data['phase'] in {'complete','rolled_back','prepared'}:return
 quarantine=inside(work/'recovered_new_files'/data['id'],root)
 for number,entry in reversed(list(enumerate(data['entries']))):
  dest=inside(entry['destination'],root);old=inside(entry['backup'],root);ready=inside(entry['prepared'],root)
  if entry['had_previous'] and old.exists():
   if dest.exists():
    quarantine.mkdir(parents=True,exist_ok=True);os.replace(dest,quarantine/f'{number:03d}')
   dest.parent.mkdir(parents=True,exist_ok=True);os.replace(old,dest)
  elif not entry['had_previous'] and not ready.exists() and dest.exists():
   quarantine.mkdir(parents=True,exist_ok=True);os.replace(dest,quarantine/f'{number:03d}')
 data['phase']='rolled_back';write_json(journal,data)

def commit_transaction(root,work,entries,backup,hook=None):
 """All replacements must be fully staged before touching existing outputs."""
 journal=work/'install_transaction.json'
 if journal.exists():
  previous=json.loads(journal.read_text(encoding='utf8'))
  if previous['phase']=='applying':raise RuntimeError('Recover the previous interrupted transaction before staging a new update')
 data={'id':backup.name,'phase':'prepared','backup':str(backup),'entries':entries}
 for e in entries:
  dest=inside(e['destination'],root);ready=inside(e['prepared'],root);old=inside(e['backup'],root)
  if not ready.exists() or old.exists():raise RuntimeError('Invalid staged replacement or occupied backup path')
  e['had_previous']=dest.exists()
 write_json(journal,data);data['phase']='applying';write_json(journal,data)
 try:
  for index,e in enumerate(entries):
   dest=Path(e['destination']);ready=Path(e['prepared']);old=Path(e['backup'])
   dest.parent.mkdir(parents=True,exist_ok=True)
   if e['had_previous']:
    old.parent.mkdir(parents=True,exist_ok=True);os.replace(dest,old)
   if hook:hook(index,'after_backup')
   os.replace(ready,dest)
   if hook:hook(index,'after_publish')
  data['phase']='complete';write_json(journal,data)
 except BaseException:
  rollback(journal,root,work)
  raise
 return {'status':'installed','revision':REVISION,'backup':str(backup),'replacement_units':len(entries)}

def install_completed_dataset(root,source,work):
 root=Path(root).resolve();source=inside(source,root);work=inside(work,root)
 audit=json.loads((work/'validation'/'final_audit.json').read_text(encoding='utf8'))
 if not (audit.get('status')=='passed' and audit.get('scope')=='complete dataset' and audit.get('images')==12500 and audit.get('videos')==25 and audit.get('frames_per_video')==500 and audit.get('pose_rows')==37500 and audit.get('resolution')==[900,600]):
  raise RuntimeError('Only a completely rendered and verified v0.9 dataset can replace old sequences')
 target=inside(root/'交付物'/OLD_NAME,root)
 if not target.exists():return {'status':'standalone_dataset','output':str(source),'reason':'No v0.8 dataset exists beside the v0.9 dataset'}
 old_lock=root/'中间结果文件'/'run_state'/'render.lock'
 try:
  with exclusive_lock(old_lock):return install_locked(root,source,target,work)
 except BlockingIOError as e:raise RuntimeError('The v0.8 renderer is still running. Stop it, then rerun run-v0.9.sh --install-only. New results remain intact.') from e

def install_locked(root,source,target,work):
 journal=work/'install_transaction.json'
 if journal.exists():rollback(journal,root,work)
 old_camera=json.loads((target/'cameras'/'camera.json').read_text(encoding='utf8'));new_camera=json.loads((source/'cameras'/'camera.json').read_text(encoding='utf8'))
 for key in ['fps','frames_per_sequence','distortion_coefficients','translation_unit','pose_direction']:
  if old_camera[key]!=new_camera[key]:raise RuntimeError(f'Existing camera convention differs: {key}; old data was not replaced')
 if (old_camera['width'],old_camera['height'],old_camera['K'])!=(1800,1200,[[2250.,0,899.5],[0,2250.,599.5],[0,0,1.]]):raise RuntimeError('Unexpected v0.8 camera calibration; no old data was changed')
 if (new_camera['width'],new_camera['height'],new_camera['K'])!=(900,600,[[1125.,0,449.5],[0,1125.,299.5],[0,0,1.]]):raise RuntimeError('Unexpected v0.9 camera calibration')
 if not same_content(source/'objects',target/'objects'):raise RuntimeError('Existing object models differ; old data was not replaced')
 pairs=[]
 for sid in range(1,6):
  scene=f'scene_{sid:02d}'
  for category in ['images','videos','poses']:pairs.append((source/category/scene/MODE,target/category/scene/MODE))
  pairs.append((source/'scenes'/scene/f'{scene}-{MODE}-v0.9-20261017.blend',target/'scenes'/scene/f'{scene}-{MODE}-v0.8-20261015.blend'))
 pairs.append((source/'cameras',target/'cameras'/MODE))
 if old_camera.get('sequence_revisions',{}).get(MODE)==REVISION and all(same_content(a,b) for a,b in pairs):
  return {'status':'already_installed','revision':REVISION,'output':str(target)}
 token='同步运动序列-替换为v0.9-'+time.strftime('%Y%m%d-%H%M%S')+'-'+str(time.time_ns()%1000000000)
 stage=inside(work/'install_staging'/token,root);stage.mkdir(parents=True)
 backup=inside(root/'交付物-backup'/token,root)
 merged=dict(old_camera);merged['sequence_revisions']={'camera_motion':'v0.8-20261015',MODE:REVISION}
 merged['geometry_note']='camera_motion retains v0.8. camera_and_objects_motion uses v0.9 independent collision-free free-space trajectories with slower camera and no fixtures.'
 merged['motion_statistics']=new_camera.get('motion_statistics',{})
 merged['default_intrinsics_motion_mode']='camera_motion'
 merged['intrinsics_by_motion_mode']={'camera_motion':'camera.json',MODE:MODE+'/camera.json'}
 merged['resolution_by_motion_mode']={'camera_motion':[1800,1200],MODE:[900,600]}
 temp_camera=stage/'updated_camera.json';write_json(temp_camera,merged);pairs.append((temp_camera,target/'cameras'/'camera.json'))
 entries=[];replacement_files={}
 for index,(src,dst) in enumerate(pairs):
  inside(src,root);inside(dst,root)
  if not src.exists():raise RuntimeError(f'Missing completed update input: {src}')
  prepared=stage/f'unit_{index:03d}';prepare_copy(src,prepared)
  entries.append({'destination':str(dst),'prepared':str(prepared),'backup':str(backup/dst.relative_to(root))})
  if prepared.is_dir():
   for p in prepared.rglob('*'):
    if p.is_file():replacement_files[(dst/p.relative_to(prepared)).relative_to(root).as_posix()]=p
  else:replacement_files[dst.relative_to(root).as_posix()]=prepared
 # Keep the older launcher's immutable-input manifest consistent with the update.
 old_manifest=root/'SHA256SUMS.txt'
 if old_manifest.exists():
  lines=[]
  for line in old_manifest.read_text(encoding='utf8').splitlines():
   digest,name=line.split('  ',1);path=PurePosixPath(name)
   if path.is_absolute() or '..' in path.parts:raise RuntimeError('Unsafe old manifest path')
   actual=inside(root/name,root)
   if not actual.is_file() or sha(actual)!=digest:raise RuntimeError(f'Old package input differs from its manifest: {name}; no old data was changed')
   lines.append(f'{sha(replacement_files[name]) if name in replacement_files else digest}  {name}')
  prepared=stage/'updated_SHA256SUMS.txt';prepared.write_text('\n'.join(lines)+'\n',encoding='utf8')
  entries.append({'destination':str(old_manifest),'prepared':str(prepared),'backup':str(backup/'SHA256SUMS.txt')})
 result=commit_transaction(root,work,entries,backup);result['output']=str(target)
 write_json(work/'install_receipt.json',result);return result
