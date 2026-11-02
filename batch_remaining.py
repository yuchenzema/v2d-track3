"""Auditable preparation of the 18 remaining Track 3 evaluation episodes."""
from pathlib import Path
import concurrent.futures, datetime, hashlib, json, urllib.request
import cv2, numpy as np
import pyarrow.parquet as pq
import xml.etree.ElementTree as ET

BASE=Path('/home/yu_eli/v2d-track3')
DATA=BASE/'datasets/video_to_data_challenge'
BATCH=BASE/'runs/evaluation_batch_20261015'
BATCH.mkdir(exist_ok=True)
ROSTER=json.loads((BASE/'v2d_submission_kit/data/track_3_evaluation_objects.json').read_text())['episodes']
EPISODES=[int(k) for k in ROSTER if int(k) not in (1,5)]
def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def audit():
    with urllib.request.urlopen('https://huggingface.co/api/datasets/nvidia/video_to_data_challenge',timeout=60) as f: info=json.load(f)
    dump(BATCH/'huggingface-revision.json',info)
    tree=[];url='https://huggingface.co/api/datasets/nvidia/video_to_data_challenge/tree/'+info['sha']+'?recursive=true&limit=1000'
    while url:
        with urllib.request.urlopen(url,timeout=60) as f:
            tree.extend(json.load(f));links=f.headers.get('Link','')
        url=next((p.split('>')[0].strip(' <') for p in links.split(',') if 'rel="next"' in p),None)
    dump(BATCH/'huggingface-full-tree.json',tree)
    resources=[]
    for item in tree:
        if item['type']!='file' or not item['path'].startswith('track_3/'):continue
        path=DATA/item['path']
        if not path.exists() or path.stat().st_size!=item['size']:
            path.parent.mkdir(parents=True,exist_ok=True)
            urllib.request.urlretrieve('https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/'+info['sha']+'/'+item['path'],path)
        record=dict(path=item['path'],size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if path.suffix in ('.json','.jsonl','.md','.urdf'):
            text=path.read_text()
            if path.suffix=='.urdf':
                root=ET.fromstring(text);record['links']=[x.attrib['name'] for x in root.findall('link')];record['meshes']=[x.attrib for x in root.findall('.//mesh')]
            elif path.suffix=='.json':record['metadata']=json.loads(text)
            elif path.suffix=='.jsonl':record['rows']=[json.loads(x) for x in text.splitlines() if x.strip()]
            else:record['text']=text
        elif path.suffix=='.parquet':
            table=pq.read_table(path);record['rows']=table.num_rows;record['columns']=table.column_names
            # Read every public value; no evaluation poses are released or used.
            for batch in table.to_batches(): batch.to_pylist()
        resources.append(record)
    dump(BATCH/'dataset-resource-audit.json',dict(revision=info['sha'],repo_files=sum(x['type']=='file' for x in tree),track3_resources=len(resources),resources=resources,attribution='NVIDIA Corporation; Track 3 assets and videos CC BY 4.0'))
    print('Hugging Face audit:',info['sha'],len(resources),'Track 3 resources',flush=True)
def prepare(ep):
    name=f'episode_{ep:06d}';run=BASE/'runs'/f'{name}_newton_batch_20261015'
    for d in ('research/source_all_frames','reconstruction','scripts','logs','videos','evaluation','models','deliverables'): (run/d).mkdir(parents=True,exist_ok=True)
    (run/'.venv-video').symlink_to(BASE/'runs/episode_000001_newton_20261013/.venv-video') if not (run/'.venv-video').exists() else None
    model=run/'models/hand_landmarker.task'
    if not model.exists():model.symlink_to(BASE/'runs/episode_000005_newton_20261015/models/hand_landmarker.task')
    source=DATA/f'track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/{name}.mp4'
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));size=[int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT)]
    thumbs=[];overview=[];ids=set(np.linspace(0,count-1,12).round().astype(int));pages=[]
    for i in range(count):
        ok,frame=cap.read();assert ok,(ep,i)
        thumb=cv2.resize(frame,(320,240));cv2.rectangle(thumb,(0,0),(230,22),(0,0,0),-1);cv2.putText(thumb,f'{name} f{i} {i/fps:.2f}s',(4,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        thumbs.append(thumb)
        if i in ids:overview.append(thumb)
        if len(thumbs)==24 or i==count-1:
            start=i-len(thumbs)+1
            while len(thumbs)<24:thumbs.append(np.zeros((240,320,3),np.uint8))
            sheet=np.concatenate([np.concatenate(thumbs[j:j+4],axis=1) for j in range(0,24,4)])
            filename=f'source_{start:06d}_{i:06d}.jpg';cv2.imwrite(str(run/'research/source_all_frames'/filename),sheet,[cv2.IMWRITE_JPEG_QUALITY,88]);pages.append(filename);thumbs=[]
    cap.release();assert len(overview)==12
    cv2.imwrite(str(run/'research/source_overview.jpg'),np.concatenate([np.concatenate(overview[j:j+4],axis=1) for j in (0,4,8)]))
    record=dict(episode=ep,sequence=name,run=str(run),objects=ROSTER[str(ep)],source=str(source),fps=fps,frames=count,resolution=size,source_review_pages=pages,all_frames_decoded=True,all_frames_visually_reviewed=False,stage='source_prepared',evaluation_ground_truth_used=False)
    dump(run/'episode-config.json',record)
    print(name,count,'frames',ROSTER[str(ep)],flush=True);return record
if __name__=='__main__':
    audit()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(prepare,EPISODES))
    dump(BATCH/'batch-manifest.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),excluded_episodes=[1,5],episodes=records,complete=False))
