"""Isolate three source-validated mesh export changes; retain all baseline poses."""
from pathlib import Path
import json,shutil,subprocess

BASE=Path('/home/yu_eli/v2d-track3')


def main():
    stage=BASE/'runs/evaluation_batch_20261015/submission_staging'
    baseline=json.loads((stage/'readiness.json').read_text())
    validated=json.loads((BASE/'submission_review_20261102/multimetric_analysis/mesh-independent-validation.json').read_text())
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
    assert len(validated)==3 and all(r['passed'] for r in validated)
    for change in validated:
        out=BASE/f"runs/evaluation_batch_20261015/mesh_ablation_{change['episode']:06d}_20261102"
        meshes=out/'meshes';out.mkdir(exist_ok=True)
        for ep,item in baseline['episodes'].items():
            for name,info in item['meshes'].items():
                target=meshes/f'episode_{int(ep):06d}/{name}.ply';target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(info['submitted_mesh'],target)
        shutil.copy2(change['candidate'],meshes/f"episode_{change['episode']:06d}/{change['object']}.ply")
        command=[str(BASE/'.venv-submission/bin/python'),str(BASE/'submission_kit_20261102/tools/pack_submission.py'),
            '--track','track_3','--results',str(stage/'results'),'--meshes',str(meshes),
            '--code-commit','https://github.com/yuchenzema/v2d-track3/commit/'+revision,
            '-o',str(out/'track3_mesh_ablation.parquet')]
        with (out/'pack.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        (out/'candidate-provenance.json').write_text(json.dumps(dict(changed_mesh=change,
            trajectories_unchanged=True,checkpoint_files_unchanged=True,code_commit=revision,
            all_six_metrics_improved=False,uploaded=False),indent=2))
        print('Prepared single-mesh ablation:',change['episode'],change['object'],flush=True)


if __name__=='__main__':main()
