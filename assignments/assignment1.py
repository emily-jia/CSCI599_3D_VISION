import trimesh
from .graph_mesh import Graph, MidVert
import numpy as np
from typing import List, Dict, Tuple


def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    verts = mesh.vertices
    faces = mesh.faces

    graph = Graph(verts, faces)
    new_verts: Dict[Tuple[int], MidVert] = dict()
    for e in graph.edges.keys():
        v1, v2 = e
        new_v = MidVert(len(new_verts), v1, v2)

        if graph.is_interior_edge(v1, v2):
            new_v.pos = (3 * verts[v1] + 3 * verts[v2]) / 8
            v3, v4 = graph.get_all_opposite_verts(v1, v2)
            new_v.pos += (verts[v3] + verts[v4]) / 8
        else:
            new_v.pos = (verts[v1] + verts[v2]) / 2
        new_verts[(v1, v2)] = new_v
    v_pos = np.array([v.pos for v in new_verts])

    even_verts = np.zeros_like(verts)
    for i, v in enumerate(graph.verts):
        n_verts = graph.get_all_neighbour_verts(i)
        n = len(n_verts)

        # boundary vertex
        if n == 2:
            beta = 3 / 4
            even_verts[i] = beta * verts[i] + (1 - beta) * (verts[n_verts[0]] + verts[n_verts[1]]) / 2
            continue

        beta = 3 / (8 * n) if n > 3 else 3 / 16
        even_verts[i] = (1 - n * beta) * verts[i] + beta * np.sum(verts[list(n_verts)], axis=0)
    
    new_verts = np.concatenate([even_verts, v_pos], axis=0)
    new_faces = []

    for f in faces:
        v1, v2, v3 = f
        v1, v2, v3 = sorted([v1, v2, v3])
        v4 = new_verts[(v1, v2)].idx + len(verts)
        v5 = new_verts[(v2, v3)].idx + len(verts)
        v6 = new_verts[(v1, v3)].idx + len(verts)
        new_faces.append([v1, v4, v6])
        new_faces.append([v4, v2, v5])
        new_faces.append([v6, v5, v3])
        new_faces.append([v4, v5, v6])
    
    mesh = trimesh.Trimesh(new_verts, new_faces)
    return mesh

def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    return mesh

if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('assets/cube.obj')
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')
    
    # apply loop subdivision over the loaded mesh
    mesh_subdivided = mesh.subdivide_loop(iterations=1)
    
    # TODO: implement your own loop subdivision here
    mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
    # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')
    
    # quadratic error mesh decimation
    # mesh_decimated = mesh.simplify_quadric_decimation(4)
    
    # TODO: implement your own quadratic error mesh decimation here
    # mesh_decimated = simplify_quadric_error(mesh, face_count=1)
    
    # print the new mesh information and save the mesh
    print(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('assets/assignment1/cube_decimated.obj')