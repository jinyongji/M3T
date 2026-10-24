"""Validate a supplied T-LESS structure-audit export and document next inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def validate(value):
    if value['status'] != 'passed_input_structure' or value['dataset'] != 'T-LESS test_primesense all':
        raise ValueError('Need a passed full test_primesense audit')
    scenes, models = value['scenes'], value['models']
    if len(scenes) != 20 or {s['scene_id'] for s in scenes} != set(range(1,21)):
        raise ValueError('Expected exactly 20 scenes')
    if len(models) != 30 or {m['obj_id'] for m in models} != set(range(1,31)) or value['model_count'] != 30:
        raise ValueError('Expected exactly 30 CAD models')
    for s in scenes:
        if s['images'] != 504 or s['rgb_sizes'] != [[720,540]]:
            raise ValueError('Unexpected RGB scope')
        for field in ['images_with_camera_extrinsics','images_with_repeated_obj_id','unique_intrinsics']:
            if not 0 <= s[field] <= s['images']:
                raise ValueError('Invalid scene count: '+field)
        if not isinstance(s['gt_instances'],int) or s['gt_instances'] < 0:
            raise ValueError('Invalid annotation count')
    if sum(s['images'] for s in scenes) != value['images'] or value['images'] != 10080:
        raise ValueError('Image totals disagree')
    if sum(s['gt_instances'] for s in scenes) != value['annotated_instances']:
        raise ValueError('Annotation totals disagree')
    for m in models:
        if not math.isfinite(m['diameter_mm']) or m['diameter_mm'] <= 0:
            raise ValueError('Invalid diameter')
        if any(not isinstance(m[k],int) or m[k]<0 for k in ['discrete_symmetries','continuous_symmetry_axes']):
            raise ValueError('Invalid symmetry metadata')
        if len(m['sha256']) != 64 or any(c not in '0123456789abcdef' for c in m['sha256']):
            raise ValueError('Invalid model hash')
    continuous = [m['obj_id'] for m in models if m['continuous_symmetry_axes']]
    discrete = [m['obj_id'] for m in models if m['discrete_symmetries']]
    none = [m['obj_id'] for m in models if not m['discrete_symmetries'] and not m['continuous_symmetry_axes']]
    geometry_only = all(m['ply_properties'] == ['property float x','property float y','property float z',
                                              'property float nx','property float ny','property float nz',
                                              'property list uchar int vertex_indices'] for m in models)
    return dict(images=value['images'], scene_count=len(scenes), model_count=len(models),
                annotated_instances=value['annotated_instances'],
                camera_extrinsics_images=sum(s['images_with_camera_extrinsics'] for s in scenes),
                per_scene_unique_intrinsics_range=[min(s['unique_intrinsics'] for s in scenes),max(s['unique_intrinsics'] for s in scenes)],
                same_class_multi_instance_scenes=[s['scene_id'] for s in scenes if s['images_with_repeated_obj_id']],
                continuous_symmetry_models=continuous, discrete_symmetry_models=discrete,
                models_without_symmetry_annotations=none,
                diameter_mm_range=[min(m['diameter_mm'] for m in models),max(m['diameter_mm'] for m in models)],
                all_cad_ply_properties_geometry_and_normals_only=geometry_only,
                tracking_runner_ready=value['tracking_runner_ready'],
                limits=['Input export consistency; no server files reread',
                        'PLY property inventory does not enumerate external texture files',
                        'No per-image K values, camera/GT transforms, persistent instance labels or view ordering in this export',
                        'Annotated-instance rows are not unique physical instances or tracking sequences'])


def render(value, findings):
    rows = ['| 场景 | 图像 | GT实例观测条目 | 不同K数量 | 有外参图像 | 同类多实例图像 |',
            '|---|---:|---:|---:|---:|---:|']
    for s in sorted(value['scenes'],key=lambda s:s['scene_id']):
        rows.append('| %02d | %d | %s | %d | %d | %d |' %
                    (s['scene_id'],s['images'],format(s['gt_instances'],','),s['unique_intrinsics'],
                     s['images_with_camera_extrinsics'],s['images_with_repeated_obj_id']))
    root = '/home/jyj/autodl-tmp/bop_datasets/tless'
    package = '/home/jyj/autodl-tmp/bop_datasets/tless_adapter_inputs-20261023.tar.gz'
    command = ('cd '+root+' && tar -czf '+package+
               ' tracking_input_audit.json models_[ce]* test_primesense/*/scene_camera.json'
               ' test_primesense/*/scene_gt.json test_primesense/*/scene_gt_info.json && ls -lh '+package)
    return '\n'.join([
        '# T-LESS服务器输入审计解读与下一步', '',
        '日期：2026-10-23。依据用户提供的 `tracking_input_audit.json`；这是服务器审计输出的本地复核，不冒充重新访问服务器或逐张解码全部图像。', '',
        '## 1. 结论', '',
        '**数据输入结构检查通过，可以继续开发五方法的T-LESS适配。报告中的 `tracking_runner_ready=false` 表示跟踪适配器尚未完成，不是下载失败。**', '',
        '| 项目 | 结果 |', '|---|---|',
        '| 场景 | 20个，每场景504张 |',
        '| RGB | 10,080张，均为720×540 |',
        '| CAD | 30个模型，审计记录了各模型SHA256 |',
        '| GT实例观测条目 | 69,552条；不能当作69,552个不同物体或跟踪序列 |',
        '| 相机外参 | 10,080张均有外参 |',
        '| 不同内参 | 每场景203–279套，所有场景均非固定K |',
        '| 同类多实例 | 9个场景存在 |',
        '| 对称性标注 | 11模型有连续对称轴，16模型有离散对称条目；3模型未标注对称性 |',
        '| CAD PLY属性 | 全部仅位置、法线、面索引；没有颜色或UV属性 |', '',
        '审计报告没有记录三归档的校验/解压日志，不单凭这个JSON追加“所有ZIP校验、完整PNG解码、depth/mask全部验证通过”的结论。已完成的输入结构审计不需要重复下载。', '',
        '## 2. 对五方法评测的直接影响', '',
        '### 2.1 相机外参完整，具备继续构建视点序列的条件', '',
        '下一步用实际 `scene_camera.json` 恢复相机中心和视角，再检查固定俯仰环及相邻视点间隔；以实际参数确定排序/切段，不从504这个数推测固定环数或顺序。', '',
        '```text',
        'C_world = -R_w2c.T @ t_w2c',
        'T_world_from_object = inverse(T_camera_from_world) @ T_camera_from_object',
        '```', '',
        '相机与目标平移都需统一从毫米转换到米。静态场景下，正确关联的同一物体在世界坐标中的位姿应一致；对称物体需保留对称变换中的旋转和平移偏置。关联应检查一一对应及歧义，不把GT列表序号直接当永久实例ID。', '',
        '### 2.2 所有场景都需要处理变化内参', '',
        '本报告只保存不同K的数量，没有具体fx/fy/cx/cy及其范围。尚不能断定只变了主点，也不能先按简单平移图像实现。读取实际K后再决定统一虚拟相机方案或逐帧更新全部投影缓存。共同图像归一化及其耗时需一致处理，不能把合成集900×600和1125焦距复制过来。', '',
        '### 2.3 9个场景有同类多实例', '',
        '场景编号为：**06、09、12、13、14、17、18、19、20**，每个这些场景的504张图均报告有重复obj_id。其余场景也可能有多个不同类别物体；“无重复obj_id”不等于单物体场景。', '',
        '实例关联与可见度规则须在运行算法前冻结，五方法使用同一份索引。报告只提供总条目数，不能据此直接给出最终序列数、初始化数或评价帧分母。', '',
        '### 2.4 对称性不是少数例外', '',
        '连续对称模型：'+', '.join('%02d'%i for i in findings['continuous_symmetry_models'])+'。', '',
        '离散对称模型：'+', '.join('%02d'%i for i in findings['discrete_symmetry_models'])+'。', '',
        '未标注对称性的模型：'+', '.join('%02d'%i for i in findings['models_without_symmetry_annotations'])+'。这里描述标注内容，不证明后者在所有观测下都完全无歧义。', '',
        '主误差应使用对称感知的MSSD/直径、MSPD及阈值曲线，并统一考虑对称性的失败/重置准则；普通5cm/5°作为附表。模型直径仅58.1257–217.156mm，固定50mm平移容差对最小件尤其宽松，不能作为精密工业任务能力的证据。具体重置阈值在小测前声明，不根据哪种方法获胜再修改。', '',
        '### 2.5 CAD-DISK在本组模型上的边界', '',
        '30个CAD的PLY属性均只有x/y/z、nx/ny/nz和面顶点索引，没有顶点颜色或UV；报告本身未枚举外部贴图文件。按保留原始几何的直接PLY→OBJ转换，无法得到当前CAD-DISK需要的有效UV/贴图输入。', '',
        '当前 `Body::has_texture()` 要求贴图路径及完整UV，CAD匹配在该判断为false时直接跳过。渲染器的灰色回退纹理不会绕过这道检查。因此应记录“配置开启但模型外观分支未参与”、有效对应数和权重，不能据网络加载成功宣称完整CAD约束已工作。视频ORB、区域、融合和时序路径仍可运行。', '',
        'T-LESS可以检验整体框架在真实弱纹理图像上的适用性和边界；若CAD-DISK没有实际参与，其结果不能独立证明该模块在真实工件上的收益。D455真实任务应另选择有准确CAD/合规外观来源的实物，保留无外观条件作为对照；不为某个方法额外贴有利纹理。', '',
        '## 3. 还缺哪些具体数据', '',
        '这份审计JSON没有包含具体相机K、外参、物体GT矩阵、可见度数值序列和CAD网格，只提供计数/属性。因此接下来需要这些原始小文件，用于本地核验视点顺序、实例关联、几何转换与对称评分；不需要把整套RGB/depth/mask下载到本机。', '',
        '在服务器复制这一整行，打包CAD/评价模型和标注：', '',
        '```bash', command, '```', '',
        '`models_[ce]*` 会包含现有models_cad以及models_eval（若存在），不包含重建模型models_reconst；其余只选相机/GT/可见度JSON与本次审计。命令写一个新的辅助包，不改已有数据或结果，不使用GPU。将生成的包下载到本机后提供给我即可。', '',
        '取得这些文件后，适配的验收顺序为：', '',
        '1. 冻结视点序列索引、同类实例一一关联和可见度统计，保留歧义/剔除原因。',
        '2. 检验逐帧K与共同图像处理，转换几何时保持模型原点、坐标轴和比例。',
        '3. 实现并验证对称误差与共同重置；逐帧保存重置前估计。',
        '4. 五方法小测，核对初始叠加、分支实际活动、计时分解，再冻结输入和运行配置做全量。', '',
        'T-LESS的真实静态视点采样补充实验、D455连续工位任务，以及现有GT-reset合成基准分别报告。当前未生成任何T-LESS跟踪成功率，也未启动全量评测。', '',
        '## 附录：20场景审计明细', '', '\n'.join(rows), ''])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args()
    raw=a.input.read_bytes(); value=json.loads(raw.decode('utf-8-sig'))
    findings=validate(value)
    a.archive.mkdir(parents=True,exist_ok=True)
    original=a.archive/'tracking_input_audit.original.json'
    if original.exists() and original.read_bytes()!=raw:
        raise ValueError('Archive already contains different input')
    original.write_bytes(raw)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(render(value,findings),encoding='utf-8')
    findings.update(status='passed_export_consistency',source_sha256=hashlib.sha256(raw).hexdigest(),
                    interpretation=str(a.output),interpretation_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    (a.archive/'interpretation.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(findings,ensure_ascii=False,indent=2))
