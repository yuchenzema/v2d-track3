"""Gravity wrench feasibility inside bounded, linearised contact friction cones."""
import numpy as np
import re
from scipy.optimize import linprog


def native_body_id(lookup,name):
    """Resolve the exact label or its MuJoCo-sanitized spelling, never another object."""
    matches={lookup(candidate) for candidate in [name,re.sub(r'[^a-zA-Z0-9_]', '_', name)]}
    matches.discard(-1)
    assert len(matches)==1,'Object body missing or ambiguous in the native contact model'
    return matches.pop()


def gravity_cone_feasible(normals,positions,friction,mass,center,gravity=(0.,0.,-9.81),normal_limit_N=25.):
    normals=np.asarray(normals,float).reshape(-1,3);positions=np.asarray(positions,float).reshape(-1,3)
    if not len(normals) or mass<=0:return False
    columns=[]
    for normal,point,mu in zip(normals,positions,friction):
        length=np.linalg.norm(normal)
        if length<.5:return False
        normal=normal/length;tangent=np.eye(3)[np.argmin(np.abs(normal))];tangent-=normal*np.dot(tangent,normal);tangent/=np.linalg.norm(tangent)
        second=np.cross(normal,tangent)
        for force in [normal+mu*tangent,normal-mu*tangent,normal+mu*second,normal-mu*second]:
            columns.append(np.r_[force,np.cross(point-np.asarray(center),force)/.1])
    A=np.array(columns).T;b=np.r_[-float(mass)*np.asarray(gravity),np.zeros(3)]
    result=linprog(np.ones(A.shape[1]),A_eq=A,b_eq=b,A_ub=np.kron(np.eye(len(normals)),np.ones((1,4))),b_ub=np.full(len(normals),normal_limit_N),bounds=(0,None),method='highs')
    return bool(result.success and np.max(np.abs(A@result.x-b))<1e-5)
