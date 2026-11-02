"""Combine a complete policy improvement with individually validated mesh variants."""
from pathlib import Path
import json,shutil,subprocess

BASE=Path(__file__).resolve().parent


def main():
    batch=BASE/'runs/evaluation_batch_20261015'
    baseline=json.loads((batch/'official_submission_20261102/original20.kaggle-receipt.json').read_text())
    motion=batch/'policy_motion_candidate_20261102'
    scored=json.loads((motion/'policy-motion.kaggle-receipt.json').read_text())
    assert all(v.get('state')=='COMPLETE' for v in scored['submissions'].values())
    out=batch/'combined_precision_motion_20261102';meshes=out/'meshes';out.mkdir(exist_ok=True)
    shutil.copytree(motion/'meshes',meshes,dirs_exist_ok=True)
    replacements=[]
    for ep in [15,32]:
        ablation=batch/f'mesh_ablation_{ep:06d}_20261102'
        score=json.loads((ablation/'cd-o.kaggle-receipt.json').read_text())['submissions']['cd_o_cm']
        assert score.get('state')=='COMPLETE'
        assert float(score['public_score'])<float(baseline['submissions']['cd_o_cm']['public_score'])
        name='wooden_spoon';source=ablation/f'meshes/episode_{ep:06d}/{name}.ply'
        shutil.copy2(source,meshes/f'episode_{ep:06d}/{name}.ply')
        replacements.append(dict(episode=ep,object=name,cd_o_ablation_submission_id=score['ref'],cd_o_ablation_score=score['public_score']))
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
    command=[str(BASE/'.venv-submission/bin/python'),str(BASE/'submission_kit_20261102/tools/pack_submission.py'),
        '--track','track_3','--results',str(motion/'results'),'--meshes',str(meshes),
        '--code-commit','https://github.com/yuchenzema/v2d-track3/commit/'+revision,
        '-o',str(out/'track3_combined_candidate.parquet')]
    with (out/'pack.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'candidate-provenance.json').write_text(json.dumps(dict(replaced_whole_episode=19,
        mesh_replacements=replacements,baseline_lid_preserved=True,
        no_posthoc_pose_edits=True,legacy_exporter_provenance_retained=True,
        all_six_metrics_improved=False,uploaded=False,code_commit=revision),indent=2))
    print('Combined candidate prepared; all six Kaggle metrics still require confirmation.')


if __name__=='__main__':main()
