"""CPU triangle silhouettes without sparse point-cloud holes or a GPU context."""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import numpy as np

_library=None
def library():
    global _library
    if _library is None:
        source=Path(__file__).with_suffix('.c');cache=source.parent/'.cache/source-mask';cache.mkdir(parents=True,exist_ok=True)
        target=cache/(hashlib.sha256(source.read_bytes()).hexdigest()+'.so')
        if not target.exists():
            temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
            subprocess.run(['cc','-O3','-shared','-fPIC',str(source),'-lm','-o',str(temporary)],check=True)
            temporary.replace(target)
        _library=ctypes.CDLL(str(target))
        _library.silhouette.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int32),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_uint8)]
        _library.silhouette.restype=None
        _library.surface_depth.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int32),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
        _library.surface_depth.restype=None
    return _library

class MeshSilhouette:
    def __init__(self,vertices,faces,intrinsics):
        self.vertices=np.asarray(vertices,dtype=float)
        self.faces=np.ascontiguousarray(faces,dtype=np.int32)
        self.k=intrinsics
        assert self.faces.ndim==2 and self.faces.shape[1]==3
        assert self.faces.min()>=0 and self.faces.max()<len(self.vertices)
    def render(self,transform):
        points=self.vertices@transform[:3,:3].T+transform[:3,3]
        pixels=np.empty_like(points);pixels[:,:2]=points[:,:2]/np.maximum(.001,points[:,2:])*[self.k['fx']/2,self.k['fy']/2]+[self.k['cx']/2,self.k['cy']/2];pixels[:,2]=points[:,2]
        pixels=np.ascontiguousarray(pixels);mask=np.zeros((384,512),np.uint8)
        library().silhouette(pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),self.faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),len(self.faces),512,384,mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return mask.astype(bool)

    def depth(self,transform):
        points=self.vertices@transform[:3,:3].T+transform[:3,3]
        pixels=np.empty_like(points)
        pixels[:,:2]=points[:,:2]/np.maximum(.001,points[:,2:])*[self.k['fx']/2,self.k['fy']/2]+[self.k['cx']/2,self.k['cy']/2]
        pixels[:,2]=points[:,2];pixels=np.ascontiguousarray(pixels)
        depth=np.zeros((384,512),np.float32)
        library().surface_depth(pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),self.faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),len(self.faces),512,384,depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return depth
