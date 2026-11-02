"""Compare an archived earlier policy before publishing a corrected training session."""
import json,shutil,traceback
from pathlib import Path
from batch_worker import context
from batch_review import choose,review

def compare(ep):
 r,c=context(ep);transition=r/'research/reference-training-session-transition.json';marker=r/'research/prior-training-session-comparison.json'
 if not transition.exists() or marker.exists() or (r/'result-report.json').exists():return
 session=json.loads(transition.read_text());selection_path=r/'final-controller-selection.json';selection=json.loads(selection_path.read_text());baseline=selection['selected'];cp='/episode/'+session['archive']+'/flash_chord_coherent_actor_transfer/'+session['old_resumable_checkpoint'];step=int(Path(cp).stem.split('_')[-1]);old_config=json.loads((r/session['archive']/'coherent-training-config.json').read_text());substeps=int(old_config['env']['sim']['substeps'])
 try:
  prior=review(ep,step,0.,session['old_reference'],substeps,checkpoint_override=cp);selected=choose([baseline,prior]);changed=selected is prior
  if changed:
   shutil.copy2(selection_path,r/'research/controller-selection-before-prior-session-comparison.json');selection['selected']=prior;selection['prior_training_session_candidate']=prior;temp=selection_path.with_suffix('.prior.tmp');temp.write_text(json.dumps(selection,indent=2));temp.replace(selection_path)
  result={'criterion':'Compare complete unassisted attempts against the same original video. Preserve the earlier policy when its native completion and whole-video source agreement qualify it as the stronger selection.','baseline':baseline,'earlier_session_candidate':prior,'selected':selected,'selected_earlier_training_session':changed,'all_earlier_weights_and_original_input_snapshots_preserved':True}
 except Exception as error:result={'selected_earlier_training_session':False,'baseline_preserved':True,'comparison_error':str(error),'traceback':traceback.format_exc()}
 marker.write_text(json.dumps(result,indent=2));print(ep,'earlier training session comparison',result.get('selected_earlier_training_session'),flush=True)
