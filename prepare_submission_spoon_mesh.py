"""Retriangulate an existing constant-thickness spoon without moving its body frame.

Dense coplanar subdivisions can stall the kit's QEM simplifier. This extracts the
existing cap boundary and triangulates the same extrusion, with at most 5 microns
of contour simplification. It does not infer a different spoon or change physics.
"""
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon


def prepare(source: Path, destination: Path):
    mesh = trimesh.load(source, force="mesh", process=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    lo, hi = vertices[:, 0].min(), vertices[:, 0].max()
    if not mesh.is_watertight or not 0.001 <= hi - lo <= 0.02:
        raise ValueError("Mesh is not the expected closed thin spoon extrusion")
    cap = faces[np.max(np.abs(vertices[faces, 0] - hi), axis=1) < 1e-8]
    edges = np.sort(np.concatenate([cap[:, [0, 1]], cap[:, [1, 2]], cap[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    adjacency = {}
    for a, b in boundary.tolist():
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("Cap boundary is not a simple closed loop")
    start = min(adjacency)
    ordered, previous, current = [start], None, start
    while True:
        following = next(n for n in adjacency[current] if n != previous)
        if following == start:
            break
        if following in ordered:
            raise ValueError("Unexpected repeated boundary vertex")
        ordered.append(following)
        previous, current = current, following
    if len(ordered) != len(adjacency):
        raise ValueError("Multiple cap loops require a separate topology-preserving exporter")
    polygon = Polygon(vertices[ordered, 1:3])
    tolerance = 5e-6
    reduced = polygon.simplify(tolerance, preserve_topology=True)
    if not polygon.is_valid or not reduced.is_valid or reduced.geom_type != "Polygon":
        raise ValueError("Spoon cap polygon is invalid")
    deviation = polygon.boundary.hausdorff_distance(reduced.boundary)
    if deviation > tolerance * 1.001:
        raise ValueError("Contour simplification exceeded its 5-micron bound")
    rebuilt = trimesh.creation.extrude_polygon(reduced, height=float(hi - lo), engine="earcut")
    points = np.asarray(rebuilt.vertices).copy()
    rebuilt.vertices = np.column_stack([points[:, 2] + lo, points[:, 0], points[:, 1]])
    original_area = float(mesh.area_faces[np.max(np.abs(vertices[faces, 0] - hi), axis=1) < 1e-8].sum())
    if abs(polygon.area - original_area) > max(1e-10, original_area * 1e-5):
        raise ValueError("Extracted boundary does not cover the original cap surface")
    bounds_error = float(np.max(np.abs(rebuilt.bounds - mesh.bounds)))
    volume_change = abs(float(rebuilt.volume) - abs(float(mesh.volume))) / abs(float(mesh.volume))
    if (len(rebuilt.vertices) > 4096 or len(rebuilt.faces) > 4096
            or not rebuilt.is_watertight or bounds_error > tolerance * 1.001
            or volume_change > 0.001):
        raise ValueError("Retriangulated mesh failed topology, size, bounds, or volume checks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.export(destination)
    return {
        "method": __doc__, "original_vertices": len(mesh.vertices),
        "original_faces": len(mesh.faces), "submitted_vertices": len(rebuilt.vertices),
        "submitted_faces": len(rebuilt.faces), "watertight": bool(rebuilt.is_watertight),
        "maximum_contour_hausdorff_error_m": float(deviation),
        "maximum_bounds_error_m": bounds_error, "relative_volume_change": volume_change,
        "body_axes_origin_scale_and_thickness_preserved": True,
        "original_physics_mesh_changed": False,
    }
