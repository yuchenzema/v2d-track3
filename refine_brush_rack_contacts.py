"""Use verified fine rack colliders, preserving visual CAD and inertial properties."""
import copy,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from improvement_common import BASE,save,utc,digest
from batch_native import _docker

R=BASE/'runs/episode_000030_newton_observed_brush_20261021'
DONOR=BASE/'runs/episode_000016_newton_stationary_hold_20261020'
REF='track3_hand_guided_brush_candidate'


def main():
    assert not (R/'coherent-input-checksums.json').exists()
    save(R/'research/improvement-status.json',{'episode':30,'status':'reconstructing','updated_utc':utc(),'run':str(R)})
    dst=R/'reconstruction/mesh_priors';src=DONOR/'reconstruction/mesh_priors';name='plastic_dish_rack'
    assert digest(dst/f'{name}.ply')==digest(src/f'{name}.ply')
    path=dst/f'{name}.urdf';backup=R/'research/original-rack.urdf';assert not backup.exists();shutil.copy2(path,backup)
    tree=ET.parse(path);link=tree.find('.//link');old_mass=ET.tostring(link.find('inertial'))
    for node in list(link.findall('collision')):link.remove(node)
    for node in ET.parse(src/f'{name}.urdf').findall('.//collision'):link.append(copy.deepcopy(node))
    for node in link.findall('collision/geometry/mesh'):
        source=src/Path(node.get('filename')).name;assert source.exists();shutil.copy2(source,dst/source.name)
        node.set('filename','/episode/reconstruction/mesh_priors/'+source.name)
    assert ET.tostring(link.find('inertial'))==old_mass;tree.write(path,encoding='unicode')
    for helper in ['native_reference_audit.py','run_geometry_cpu.py','project_native_object_contacts.py','diagnose_static_grasp.py']:
        shutil.copy2(BASE/helper,R/'scripts'/helper)
    before=R/'research/hand-guided-before-contact-projection.parquet';motion=next((R/'human_motion_data'/REF).rglob('motion.parquet'));shutil.copy2(motion,before)
    container=json.loads((R/'episode-config.json').read_text())['native_container']
    def cpu(script,env,label):
        _docker(['exec',container,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',*env,'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/'+script],R/f'logs/{label}.log')
    cpu('project_native_object_contacts.py',[f'V2D_PROJECT_REFERENCE={REF}','V2D_PROJECTION_MOBILE_OBJECTS=[0]','V2D_PROJECTION_BOUND_M=0.04'],'fine-rack-contact-projection')
    cpu('native_reference_audit.py',[f'V2D_AUDIT_REFERENCE={REF}'],'fine-rack-reference-audit')
    save(R/'research/fine-rack-collider-provenance.json',{'updated_utc':utc(),'donor_run':str(DONOR),'visual_cad_sha256':digest(dst/f'{name}.ply'),
         'collision_parts':len(link.findall('collision')),'visual_mesh_and_inertial_properties_preserved':True,
         'original_urdf_backup':str(backup.relative_to(R)),'projection_is_reference_reconstruction_only':True,
         'training_or_evaluation_object_assistance':0.})
    print('Fine rack geometry and bounded object-contact projection checked.',flush=True)


if __name__=='__main__':main()
