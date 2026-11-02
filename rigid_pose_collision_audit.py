"""Query native collision solids at recorded rigid poses, without stepping physics."""
import hashlib
import json
import re
import numpy as np
from scipy.spatial.transform import Rotation

NUMERICAL_EPSILON_M = 1e-6
REQUIRED_CATEGORIES = {'hand_object', 'object_object', 'object_environment', 'hand_environment', 'hand_hand'}


def contact_retry_thresholds(strict):
    # Rescue proposals must trigger at the same tolerance as the strict gate,
    # including millimetre contacts below the historical 3/5 mm heuristics.
    return (NUMERICAL_EPSILON_M,NUMERICAL_EPSILON_M) if strict else (.003,.005)


def unfitted_blend_frames(first,second,fits,count):
    seam=second['held_interval'][0]
    lo=max(first['held_interval'][0],seam-3);hi=min(second['held_interval'][1],seam+3)
    protected={i for fit in fits for i in range(fit['held_interval'][0],fit['held_interval'][1]+1)}
    return lo,hi,[i for i in range(max(0,lo),min(count-1,hi)+1) if i not in protected]


def normalized(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def category(first, second):
    if first == second:
        return None  # Pieces of one rigid solid and each hand's joint assembly.
    kinds = sorted([first.split(':')[0], second.split(':')[0]])
    return {'hand,object': 'hand_object', 'object,object': 'object_object',
            'environment,object': 'object_environment', 'environment,hand': 'hand_environment',
            'hand,hand': 'hand_hand'}.get(','.join(kinds))


def hand_clearance_penalties(model, data, hand_geoms, hand_bodies, clearance_m=.0005):
    """Include filtered hand/table pairs when fitting a nonintersecting reference."""
    import mujoco
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    active=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
    others=np.flatnonzero(active & ~np.isin(model.geom_bodyid,list(hand_bodies)))
    penalties=np.zeros(len(hand_geoms));maximum=0.
    if not len(others):return penalties,maximum
    plane= model.geom_type[others]==int(mujoco.mjtGeom.mjGEOM_PLANE)
    for index,geom in enumerate(hand_geoms):
        if not active[geom]:continue
        nearby=plane | (np.linalg.norm(data.geom_xpos[others]-data.geom_xpos[geom],axis=1)
                       <=model.geom_rbound[others]+model.geom_rbound[geom]+.002)
        for other in others[nearby]:
            distance=float(mujoco.mj_geomDistance(model,data,int(geom),int(other),.002,None))
            assert np.isfinite(distance)
            maximum=max(maximum,-distance)
            penalties[index]=max(penalties[index],clearance_m-distance)
    return penalties,maximum


class PoseCollisionAuditor:
    def __init__(self, model, data, body_names, template_body_poses, objects):
        import mujoco
        self.model, self.data = model, data
        # Select accurate native CCD for independent distance queries only.
        # This checker never advances or changes the recorded simulation state.
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.body_names = list(body_names)
        template = np.asarray(template_body_poses, dtype=float)
        assert template.shape == (len(body_names), 7) and np.isfinite(template).all()
        lookup = {normalized(name): i for i, name in enumerate(body_names)}
        assert len(lookup) == len(body_names), 'Ambiguous native body names'
        rotations = Rotation.from_quat(template[:, 3:]).as_matrix()
        self.geom_ids = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        self.records = []
        groups = {}
        for geom in self.geom_ids:
            body = int(model.geom_bodyid[geom])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or 'world'
            index = lookup.get(normalized(name))
            if index is None:
                assert body == 0 or int(model.body_weldid[body]) == 0, f'Unmapped moving collision body: {name}'
                group = 'environment:static'
                local_pos = data.geom_xpos[geom].copy()
                local_rot = data.geom_xmat[geom].reshape(3, 3).copy()
            else:
                label = body_names[index]
                obj = next((obj for obj in objects if normalized(label) == normalized(f'track3_{obj}/object')), None)
                if obj:
                    group = 'object:' + obj
                elif label.startswith('left_'):
                    group = 'hand:left'
                elif label.startswith('right_'):
                    group = 'hand:right'
                else:
                    assert int(model.body_weldid[body]) == 0 or label in {'support', 'table', 'ground', 'floor'}, f'Unknown collision group: {label}'
                    group = 'environment:static'
                local_pos = rotations[index].T @ (data.geom_xpos[geom] - template[index, :3])
                local_rot = rotations[index].T @ data.geom_xmat[geom].reshape(3, 3)
            self.records.append((int(geom), index, local_pos, local_rot, name, group))
            groups[group] = groups.get(group, 0) + 1
        for group in ['hand:left', 'hand:right', 'environment:static', *['object:' + obj for obj in objects]]:
            assert groups.get(group, 0) > 0, f'Missing collision solids for {group}'
        self.groups = groups
        self.pairs = []
        for i, first in enumerate(self.records):
            for j in range(i + 1, len(self.records)):
                second = self.records[j]
                kind = category(first[5], second[5])
                if kind:
                    self.pairs.append((i, j, kind))
        self.categories = {kind for _, _, kind in self.pairs}
        required = REQUIRED_CATEGORIES if len(objects) > 1 else REQUIRED_CATEGORIES - {'object_object'}
        assert required <= self.categories, 'Collision-pair coverage incomplete'
        geometry = hashlib.sha256()
        for value in [model.geom_type, model.geom_size, model.geom_rbound, model.mesh_vert,
                      model.mesh_face, model.geom_dataid]:
            geometry.update(np.asarray(value).tobytes())
        for record in self.records:
            geometry.update(json.dumps([record[0], record[1], record[4], record[5]]).encode())
            geometry.update(record[2].tobytes()); geometry.update(record[3].tobytes())
        self.geometry_sha256 = geometry.hexdigest()

    def inspect(self, body_poses, focus_group=None):
        import mujoco
        poses = np.asarray(body_poses, dtype=float)
        assert poses.shape == (len(self.body_names), 7) and np.isfinite(poses).all()
        rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
        for geom, index, local_pos, local_rot, _, _ in self.records:
            if index is None:
                self.data.geom_xpos[geom] = local_pos
                self.data.geom_xmat[geom] = local_rot.reshape(9)
            else:
                self.data.geom_xpos[geom] = rotations[index] @ local_pos + poses[index, :3]
                self.data.geom_xmat[geom] = (rotations[index] @ local_rot).reshape(9)
        maximum = 0.
        near_queries = 0
        penetrations = []
        by_category = {kind: 0. for kind in self.categories}
        pairs_covered = 0
        for i, j, kind in self.pairs:
            first, second = self.records[i], self.records[j]
            if focus_group is not None and focus_group not in [first[5], second[5]]:
                continue
            pairs_covered += 1
            a, b = first[0], second[0]
            plane = (int(self.model.geom_type[a]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                     or int(self.model.geom_type[b]) == int(mujoco.mjtGeom.mjGEOM_PLANE))
            if not plane:
                radii = float(self.model.geom_rbound[a] + self.model.geom_rbound[b])
                if np.linalg.norm(self.data.geom_xpos[a] - self.data.geom_xpos[b]) > radii + .001:
                    continue  # Conservative bounding spheres; cannot omit an overlap.
            near_queries += 1
            distance = float(mujoco.mj_geomDistance(self.model, self.data, a, b, .001, None))
            assert np.isfinite(distance), 'Nonfinite collision distance'
            depth = max(0., -distance)
            maximum = max(maximum, depth)
            by_category[kind] = max(by_category[kind], depth)
            if depth > NUMERICAL_EPSILON_M:
                penetrations.append({'bodies': [first[4], second[4]], 'groups': [first[5], second[5]],
                                     'geom_ids': [a, b], 'category': kind, 'penetration_m': depth})
        penetrations.sort(key=lambda hit: -hit['penetration_m'])
        return {'maximum_penetration_m': maximum, 'maximum_by_category_m': by_category,
                'penetrating_pair_count': len(penetrations), 'worst_contacts': penetrations[:12],
                'pairs_covered': pairs_covered, 'narrow_phase_queries': near_queries}
