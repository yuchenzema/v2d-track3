"""Read and verify every Track 3 resource against a pinned official HF inventory."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pyarrow.parquet as pq
import trimesh
from improvement_common import BASE,BATCH,save,utc

DATA=BASE/'datasets/video_to_data_challenge'
OUT=BATCH/'focused_episodes_20261028/huggingface_audit_20261028'


def main():
    inventory=json.loads((OUT/'track3-inventory.json').read_text())
    files=[entry for entry in inventory['entries'] if entry['type']=='file']
    report={'updated_utc':utc(),'repository':'nvidia/video_to_data_challenge','revision':inventory['revision'],
            'official_file_count':len(files),'files':[],'structured_resources':{},'urdf':{},'meshes':{},'parquets':{},'videos':{},'errors':[]}
    for index,entry in enumerate(files):
        path=DATA/entry['path'];row={'path':entry['path'],'expected_size':entry['size'],'exists':path.exists()}
        if path.exists():
            content=path.read_bytes();sha256=hashlib.sha256(content).hexdigest();git=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            expected=(entry.get('lfs') or {}).get('oid') or entry['oid']
            actual=sha256 if entry.get('lfs') else git
            row.update(size=len(content),sha256=sha256,expected_digest=expected,digest_kind='sha256' if entry.get('lfs') else 'git_blob_sha1',verified=actual==expected and len(content)==entry['size'])
        else:row['verified']=False
        if not row['verified']:report['errors'].append({'path':entry['path'],'error':'Missing or differs from pinned official repository'})
        report['files'].append(row)
        save(OUT/'progress.json',{'updated_utc':utc(),'stage':'verifying_all_files','files_checked':index+1,'total_files':len(files),'videos_decoded':0})
    assert not report['errors'],report['errors']
    for entry in files:
        path=DATA/entry['path'];suffix=path.suffix
        if suffix in {'.md','.json','.jsonl','.urdf'} or path.name=='.gitignore':
            text=path.read_text()
            if suffix=='.json':value=json.loads(text)
            elif suffix=='.jsonl':value=[json.loads(line) for line in text.splitlines() if line.strip()]
            else:value=text
            report['structured_resources'][entry['path']]=value
        if suffix=='.urdf':
            root=ET.parse(path).getroot();meshes=[]
            for node in root.findall('.//mesh'):
                target=(path.parent/node.get('filename')).resolve();assert target.is_relative_to(DATA) and target.exists()
                meshes.append({'path':str(target.relative_to(DATA)),'scale':node.get('scale','1 1 1')})
            mass=root.find('.//inertial/mass');inertia=root.find('.//inertial/inertia')
            report['urdf'][path.stem]={'robot':root.get('name'),'mass_kg':float(mass.get('value')) if mass is not None else None,
                'mass_is_measured':False,'inertia':inertia.attrib if inertia is not None else None,'mesh_references':meshes,
                'marker_centers_explicitly_annotated':False}
        if suffix=='.glb':
            scene=trimesh.load(path,force='scene',process=False);geometry=scene.to_geometry()
            report['meshes'][entry['path']]={'bounds_m':geometry.bounds.tolist(),'vertices':len(geometry.vertices),'faces':len(geometry.faces),
                'nodes':list(scene.graph.nodes_geometry),'geometry_names':list(scene.geometry),'metadata':scene.metadata,
                'geometry_metadata':{name:g.metadata for name,g in scene.geometry.items()},'texture_present':geometry.visual.kind=='texture'}
        if suffix=='.parquet':
            table=pq.read_table(path);rows=table.to_pylist();meta={k.decode():v.decode(errors='replace') for k,v in (table.schema.metadata or {}).items()}
            episode=int(rows[0]['episode_index']);objects=[o['name'] for o in rows[0]['observation.objects']]
            invalid=[];invisible=0;visible=0;norms=[]
            for frame,row in enumerate(rows):
                if row['frame_index']!=frame or row['episode_index']!=episode:invalid.append([frame,'frame_or_episode_index'])
                if [o['name'] for o in row['observation.objects']]!=objects:invalid.append([frame,'object_membership'])
                for obj in row['observation.objects']:
                    pose=np.asarray(obj['pose'],float)
                    if obj['visible']:
                        visible+=1;norms.append(float(np.linalg.norm(pose[3:])))
                        if not np.isfinite(pose).all() or abs(norms[-1]-1)>.002:invalid.append([frame,obj['name'],'invalid_visible_pose'])
                    else:
                        invisible+=1
                        if np.any(pose):invalid.append([frame,obj['name'],'nonzero_invisible_pose'])
            times=np.asarray([row['timestamp'] for row in rows]);capture=np.asarray([row['capture_time'] for row in rows]);diff=np.diff(times)
            report['parquets'][str(episode)]={'path':entry['path'],'rows':len(rows),'columns':table.column_names,'schema_metadata':meta,'objects':objects,
                'visible_object_samples':visible,'invisible_object_samples':invisible,'validation_errors':invalid,
                'timestamp_start_end':times[[0,-1]].tolist(),'capture_time_start_end':capture[[0,-1]].tolist(),
                'timestamp_delta_min_median_max':np.quantile(diff,[0,.5,1]).tolist() if len(diff) else [],
                'quaternion_norm_min_max':[min(norms),max(norms)] if norms else [],'pose_order':'x y z qw qx qy qz','pose_frame':'world_T_scanned_mesh'}
            if invalid:report['errors'].append({'path':entry['path'],'error':'Public pose table inconsistencies','details':invalid[:10]})
    videos=[entry for entry in files if entry['path'].endswith('.mp4')]
    for index,entry in enumerate(videos):
        path=DATA/entry['path']
        result=subprocess.run(['ffprobe','-v','error','-threads','2','-count_frames','-select_streams','v:0',
            '-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
        stream=json.loads(result.stdout)['streams'][0];count=int(stream['nb_read_frames']);episode=int(path.stem.split('_')[-1]);public='/public/' in entry['path']
        if public and count!=report['parquets'][str(episode)]['rows']:report['errors'].append({'path':entry['path'],'error':'Video/parquet frame counts differ'})
        report['videos'][entry['path']]={**stream,'decoded_frames':count,'episode':episode,'split':'public' if public else 'evaluation'}
        save(OUT/'progress.json',{'updated_utc':utc(),'stage':'decoding_every_video','files_checked':len(files),'total_files':len(files),'videos_decoded':index+1,'total_videos':len(videos)})
        if index%15==0:print('videos fully decoded',index+1,'/',len(videos),flush=True)
    public_info=report['structured_resources']['track_3/public/meta/info.json']
    assert sum(v['rows'] for v in report['parquets'].values())==public_info['total_frames']
    report.update(updated_utc=utc(),files_verified=sum(row['verified'] for row in report['files']),
        total_bytes=sum(row['size'] for row in report['files']),public_pose_rows=sum(v['rows'] for v in report['parquets'].values()),
        all_files_read_and_verified=True,all_videos_fully_decoded=len(report['videos'])==len(videos),
        explicit_marker_coordinate_files_found=False,complete=not report['errors'],
        scope='Every file in the pinned track_3 subtree; root dataset card reviewed separately. Video decoding validates every frame but is not a claim of manual visual review of every frame.',
        important_constraints=['Public and evaluation episode IDs are different. Public poses are development references, not evaluation ground truth.',
            'Visible=false poses contain zero quaternions and must be excluded.',
            'Poses use metres, w-first quaternions and the scanned mesh frame; XY world origins are centred independently per episode.',
            'Factory camera calibration includes stereo baselines but no camera-to-mocap-world extrinsic.',
            'URDF mass/inertia are placeholders/estimates, not measured physical parameters.',
            'Marker centres must be derived and validated from mesh/video observations; no separate annotated marker-centre table is released.'])
    save(OUT/'resource-report.json',report)
    summary={key:report[key] for key in ['updated_utc','revision','official_file_count','files_verified','total_bytes','public_pose_rows','all_files_read_and_verified','all_videos_fully_decoded','complete','errors','important_constraints']}
    summary.update(video_count=len(videos),mesh_count=len(report['meshes']),urdf_count=len(report['urdf']),parquet_count=len(report['parquets']))
    save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
