import trimesh
from graph_mesh import Graph, MidVert, Edge
import numpy as np
from typing import List, Dict, Tuple
import heapq


def _subdivision_step(mesh):
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
    v_pos = np.array([v.pos for v in new_verts.values()])

    even_verts = np.zeros_like(verts)
    for i, v in enumerate(graph.verts):
        n_verts = graph.get_all_neighbour_verts(i)
        n = len(n_verts)

        # boundary vertex
        if n == 2:
            beta = 3 / 4
            even_verts[i] = beta * verts[i] + (1 - beta) * (verts[n_verts[0]] + verts[n_verts[1]]) / 2
            continue

        # beta = 3 / (8 * n) if n > 3 else 3 / 16
        beta = 1/n * (5/8 - (3/8 + 1/4 * np.cos(2 * np.pi / n))**2)
        even_verts[i] = (1 - n * beta) * verts[i] + beta * np.sum(verts[list(n_verts)], axis=0)
    
    new_verts_p = np.concatenate([even_verts, v_pos], axis=0)
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
    
    mesh = trimesh.Trimesh(new_verts_p, new_faces)
    return mesh


def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    for i in range(iterations):
        mesh = _subdivision_step(mesh)
    return mesh


def get_K(v1, v2, v3):
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    a, b, c = n[0], n[1], n[2]
    d = -np.dot(n, v1)
    K = np.array([a**2, a*b, a*c, a*d, b**2, b*c, b*d, c**2, c*d, d**2])
    return K


def get_min_err(K):
    orig_k_mat = np.array([[K[0], K[1], K[2], K[3]],
                      [K[1], K[4], K[5], K[6]],
                      [K[2], K[5], K[7], K[8]],
                      [K[3], K[6], K[8], K[9]]])
    k_mat = orig_k_mat.copy()
    k_mat[3, 3] = 1
    k_mat[3, :3] = 0
    min_v = np.linalg.inv(k_mat) @ np.array([[0, 0, 0, 1]]).T # (4, 1)
    min_err = min_v.T @ orig_k_mat @ min_v
    min_err = min_err[0, 0]
    return min_v, min_err


def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    verts = mesh.vertices
    faces = mesh.faces

    graph = Graph(verts, faces)
    face_dict = dict()
    for i in range(len(faces)):
        vid1, vid2, vid3 = faces[i]
        v1, v2, v3 = verts[vid1], verts[vid2], verts[vid3]
        K = get_K(v1, v2, v3)
        graph.verts[vid1].add_face_quadric(K)
        graph.verts[vid2].add_face_quadric(K)
        graph.verts[vid3].add_face_quadric(K)

        face_dict[vid1] = face_dict.get(vid1, []) + [i]
        face_dict[vid2] = face_dict.get(vid2, []) + [i]
        face_dict[vid3] = face_dict.get(vid3, []) + [i]
    
    for vid1, vid2 in graph.edges.keys():
        if graph.is_interior_edge(vid1, vid2):
            v1, v2 = graph.verts[vid1], graph.verts[vid2]
            e = graph.edges[(vid1, vid2)]
            K = v1.quadrics + v2.quadrics
            e.quadrics = K
            e.min_vert, e.min_err = get_min_err(K)
        else:
            e = graph.edges[(vid1, vid2)]
            e.min_err = float('inf')

    heap = list(graph.edges.values())
    cur_faces = len(graph.faces)
    heapq.heapify(heap)
    while cur_faces > face_count:
        e = heapq.heappop(heap)
        if e.quadrics is None: # no longer valid after collapse
            continue
        vid1, vid2 = e.v1, e.v2
        v1, v2 = graph.verts[vid1], graph.verts[vid2]
        if v1.pos is None or v2.pos is None:  # already collapsed
            continue

        v1.pos = e.min_vert
        v2.pos = None
        v1.quadrics = e.quadrics
        v2.quadrics = None

        v1.adj_vs.remove(vid2)
        print("pop3", vid1, vid2)
        graph.edges.pop((vid1, vid2) if vid1 < vid2 else (vid2, vid1))
        remove_vs = list()
        # collapse the edge
        del_faces = []
        for fid in face_dict[vid1]:
            if vid2 in graph.faces[fid]:
                cur_faces -= 1
                del_faces.append(fid)

                other_v = [v for v in graph.faces[fid] if v != vid2 and v != vid1][0]
                remove_vs.append(other_v)
                print("pop2", vid2, other_v)
                graph.edges.pop((vid2, other_v) if vid2 < other_v else (other_v, vid2))
                graph.verts[other_v].adj_vs.remove(vid2)
                face_dict[other_v].remove(fid)
        for fid in del_faces:
            face_dict[vid1].remove(fid)
            
        for vid in v1.adj_vs:
            key = (vid1, vid) if vid1 < vid else (vid, vid1)
            prev_e = graph.edges[key]
            prev_e.quadrics = None

            new_e = Edge(key[0], key[1])
            new_e.quadrics = v1.quadrics + graph.verts[vid].quadrics
            new_e.min_vert, new_e.min_err = get_min_err(new_e.quadrics)
            graph.edges[key] = new_e
            heapq.heappush(heap, new_e)

        new_adj = set()
        for fid in face_dict[vid2]:
            f = graph.faces[fid]
            if vid1 in f:
                continue
            
            other_vs = [v for v in f if v != vid2 and v not in remove_vs]
            for vid in other_vs:
                new_adj.add(vid)
                
            face_dict[vid1].append(fid)
            for i, vid in enumerate(f):
                if vid == vid2:
                    graph.faces[fid][i] = vid1
        
        v1.adj_vs = v1.adj_vs.union(new_adj)
        for vid in new_adj:
            key1 = (vid2, vid) if vid2 < vid else (vid, vid2)
            print("pop4", vid2, vid)
            graph.edges.pop(key1)
            graph.verts[vid].adj_vs.remove(vid2)
            graph.verts[vid].adj_vs.add(vid1)

            key2 = (vid1, vid) if vid1 < vid else (vid, vid1)
            assert key2 not in graph.edges, f"Warning: edge already exists, {vid1}, {vid}"
            print("new edge2", vid1, vid)
            e = Edge(key2[0], key2[1])
            e.quadrics = v1.quadrics + graph.verts[vid].quadrics
            e.min_vert, e.min_err = get_min_err(e.quadrics)
            graph.edges[key2] = e
            heapq.heappush(heap, e)
        face_dict.pop(vid2)

    verts = np.array([v.pos for v in graph.verts if v.pos is not None])
    mask_pop = np.array([v.pos is not None for v in graph.verts], dtype=bool)
    new_id = mask_pop.astype(int).cumsum() - mask_pop
    faces = np.array([[new_id[i] for i in f if mask_pop[i]] for f in graph.faces]) # TODO

    mesh = trimesh.Trimesh(verts, faces)
    return mesh

if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('assets/cube.obj')
    # mesh = trimesh.creation.box(extents=[1, 1, 1])
    # print(f'Mesh Info: {mesh}')
    
    # # apply loop subdivision over the loaded mesh
    # mesh_subdivided = mesh.subdivide_loop(iterations=1)
    # mesh_subdivided.export('assets/assignment1/gt_1.obj')
    
    # # TODO: implement your own loop subdivision here
    # for i in range(1, 5):
    #     mesh_subdivided = subdivision_loop(mesh, iterations=i)
    
    #     # print the new mesh information and save the mesh
    #     print(f'Subdivided Mesh Info: {mesh_subdivided}')
    #     mesh_subdivided.export(f'assets/assignment1/cube_subdivided_{i}.obj')

    # mesh = trimesh.creation.icosahedron()
    # print(f'Mesh Info: {mesh}')
    # for i in range(1, 5):
    #     mesh_subdivided = subdivision_loop(mesh, iterations=i)
    #     print(f'Subdivided Mesh Info: {mesh_subdivided}')
    #     mesh_subdivided.export(f'assets/assignment1/icosahedron_subdivided_{i}.obj')
        
    # quadratic error mesh decimation
    mesh = trimesh.load('assets/assignment1/icosahedron_subdivided_4.obj')
    # mesh_decimated = mesh.simplify_quadric_decimation(1024)
    # mesh_decimated.export('assets/assignment1/bunny_decimated_gt.obj')
    
    # TODO: implement your own quadratic error mesh decimation here
    mesh_decimated = simplify_quadric_error(mesh, face_count=12)
    
    # print the new mesh information and save the mesh
    print(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('assets/assignment1/test.obj')