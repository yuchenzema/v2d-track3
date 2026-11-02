"""Sparse texture-to-video photometric residual in the scanned mesh body frame."""
import cv2
import numpy as np
import trimesh


class MeshPhotometricReference:
    def __init__(self,points,colors):
        self.points=np.asarray(points,float);self.colors=np.asarray(colors,float)

    @classmethod
    def from_mesh(cls,path,count=1200):
        mesh=trimesh.load(path,force='scene',process=False).to_geometry()
        if getattr(mesh.visual,'kind',None) not in ['texture','vertex','face']:return None
        visual=mesh.visual.to_color() if mesh.visual.kind=='texture' else mesh.visual
        colors=visual.vertex_colors[:,:3].astype(float)/255.
        if float(np.mean(np.std(colors,axis=0)))<.015:return None
        points,faces=trimesh.sample.sample_surface(mesh,count,seed=81731)
        triangles=mesh.triangles[faces]
        barycentric=trimesh.triangles.points_to_barycentric(triangles,points)
        sampled=np.einsum('ni,nij->nj',barycentric,colors[mesh.faces[faces]])
        return cls(points,sampled)

    def residual(self,transform,image_rgb,intrinsics,surface_depth,visible):
        points=self.points@transform[:3,:3].T+transform[:3,3]
        pixels=points[:,:2]/np.maximum(points[:,2:],.001)*[intrinsics['fx']/2,intrinsics['fy']/2]+[intrinsics['cx']/2,intrinsics['cy']/2]
        finite=np.isfinite(pixels).all(axis=1)&np.isfinite(points[:,2])
        bounded=finite&(points[:,2]>.05)&(pixels[:,0]>=0)&(pixels[:,0]<image_rgb.shape[1]-1)&(pixels[:,1]>=0)&(pixels[:,1]<image_rgb.shape[0]-1)
        ids=np.flatnonzero(bounded)
        if len(ids)<30:return 0.,0
        uv=pixels[ids];u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int)
        good=visible[v,u]&(surface_depth[v,u]>.05)&(np.abs(surface_depth[v,u]-points[ids,2])<.006)
        ids=ids[good];uv=uv[good]
        if len(ids)<30:return 0.,len(ids)
        sampled=cv2.remap(image_rgb,uv[:,0].astype(np.float32)[None],uv[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0]
        expected=self.colors[ids]
        # Normalise exposure/white balance per visible subset. Constant-colour
        # surfaces contribute no invented texture signal.
        contrast=np.mean(np.std(expected,axis=0))
        if contrast<.015:return 0.,len(ids)
        a=(sampled-sampled.mean(0))/np.maximum(sampled.std(0),.03)
        b=(expected-expected.mean(0))/np.maximum(expected.std(0),.03)
        error=np.mean((a-b)**2,axis=1)
        weight=.1+np.linalg.norm(expected-expected.mean(0),axis=1)
        loss=float(np.average(np.sqrt(1+np.minimum(error,9.))-1,weights=weight))
        return loss,len(ids)
