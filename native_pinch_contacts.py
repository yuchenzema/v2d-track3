"""Bounded reference pinch fitting against native collision surfaces.

These geometry queries do not apply forces or advance physics. A feasible
gravity cone is a diagnostic; an unassisted dynamic grasp must still pass.
"""
import numpy as np

from grasp_physics_checks import gravity_cone_feasible


def valid_distance_witness(distance, points, maximum_distance):
    points = np.asarray(points, dtype=float)
    return bool(np.isfinite(distance) and points.shape == (6,) and np.isfinite(points).all()
                and distance < maximum_distance
                and abs(np.linalg.norm(points[3:] - points[:3]) - abs(distance)) < 1e-5)


class NativePinchContacts:
    def __init__(self, model, data, hand_geoms, object_body, active_fingers):
        import mujoco
        self.model, self.data, self.body = model, data, object_body
        self.pairs = []
        colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        objects = np.flatnonzero(colliding & (model.geom_bodyid == object_body))
        found = {}
        for finger, active in zip(['thumb', 'index', 'middle', 'ring', 'pinky'], active_fingers):
            if not active:
                continue
            best = None
            for hand in hand_geoms:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[hand])) or ''
                if not colliding[hand] or f'_{finger}_' not in name:
                    continue
                for obj in objects:
                    if np.linalg.norm(data.geom_xpos[hand] - data.geom_xpos[obj]) > (
                            model.geom_rbound[hand] + model.geom_rbound[obj] + .025):
                        continue
                    points = np.zeros(6)
                    distance = float(mujoco.mj_geomDistance(model, data, int(hand), int(obj), .025, points))
                    if valid_distance_witness(distance, points, .025) and (best is None or distance < best[0]):
                        best = (distance, int(hand), int(obj))
            if best is not None:
                found[finger] = best
        # Preserve the observed thumb-versus-fingers grasp. Do not invent a
        # pinch for a support palm, a released hand or a distant object.
        others = [(value[0], finger, value) for finger, value in found.items() if finger != 'thumb']
        if 'thumb' in found and others:
            self.pairs = [('thumb', *found['thumb'][1:])]
            # A two-point pinch can leave the long tool's gravity torque
            # unsupported. Retain nearby observed fingers as additional sites.
            self.pairs += [(finger, *value[1:]) for gap, finger, value in sorted(others) if gap <= .01]

    def query(self):
        import mujoco
        rows = []
        for finger, hand, obj in self.pairs:
            points = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(self.model, self.data, hand, obj, .05, points))
            valid = valid_distance_witness(distance, points, .05)
            normal = ((points[3:] - points[:3]) / distance if valid and abs(distance) > 1e-9
                      else np.zeros(3))
            position=points[3:]
            if valid and np.linalg.norm(normal)<.5:
                # At exact touching, coincident GJK witnesses cannot define a
                # direction. Use the actual native collision manifold's normal
                # for this exact pair, rather than inventing or omitting it.
                for contact in self.data.contact:
                    a,b=map(int,contact.geom)
                    if (a,b)==(hand,obj) or (a,b)==(obj,hand):
                        direction=np.asarray(contact.frame).reshape(-1)[:3]
                        normal=direction if a==hand else -direction
                        position=np.asarray(contact.pos)
                        break
            rows.append(dict(finger=finger, hand_geom=hand, object_geom=obj,
                             distance_m=distance if valid else .05, valid_witness=valid,
                             position=position.tolist(), normal_on_object=normal.tolist(),
                             friction=float(min(self.model.geom_friction[hand, 0], self.model.geom_friction[obj, 0]))))
        return rows

    def residual(self):
        # Keep 0.5 mm clearance, identical to the strict reference fitter.
        return np.array([(row['distance_m'] - .0005) / .0005 for row in self.query()])

    def support_residual(self):
        """Fitting objective only: bounded friction-cone wrench imbalance.

        No force is applied to the model. The original exact feasibility and
        penetration acceptance still decide whether a proposed fit is usable.
        """
        from scipy.optimize import lsq_linear
        gravity=np.asarray(self.model.opt.gravity,dtype=float)
        mass=float(self.model.body_mass[self.body]);center=np.asarray(self.data.xipos[self.body])
        target=np.r_[-mass*gravity,np.zeros(3)];columns=[]
        for row in self.query():
            normal=np.asarray(row['normal_on_object']);length=np.linalg.norm(normal)
            if not row['valid_witness'] or length<.5:continue
            normal=normal/length;tangent=np.eye(3)[np.argmin(np.abs(normal))]
            tangent-=normal*np.dot(tangent,normal);tangent/=np.linalg.norm(tangent)
            second=np.cross(normal,tangent);mu=row['friction'];point=np.asarray(row['position'])
            for force in [normal+mu*tangent,normal-mu*tangent,normal+mu*second,normal-mu*second]:
                columns.append(np.r_[force,np.cross(point-center,force)/.1])
        scale=max(.1,mass*np.linalg.norm(gravity))
        if not columns:return -4*target/scale
        matrix=np.asarray(columns).T
        # Four nonnegative edge loads capped at 25/4 N preserve the existing
        # 25 N per-contact bound. Final acceptance uses the exact grouped LP.
        solution=lsq_linear(matrix,target,bounds=(0.,25./4),lsq_solver='exact',tol=1e-6,max_iter=40)
        return 4*(matrix@solution.x-target)/scale

    def summary(self):
        rows = self.query()
        near = [row for row in rows if row['valid_witness'] and -1e-6 <= row['distance_m'] <= .001
                and np.linalg.norm(row['normal_on_object']) > .5]
        feasible = gravity_cone_feasible([r['normal_on_object'] for r in near], [r['position'] for r in near],
                                        [r['friction'] for r in near], self.model.body_mass[self.body],
                                        self.data.xipos[self.body], gravity=self.model.opt.gravity)
        return dict(contacts=rows, reference_gravity_cone_feasible=bool(feasible),
                    diagnostic_only=True, dynamic_grasp_verified=False)
