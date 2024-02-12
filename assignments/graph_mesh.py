import numpy as np
from typing import List, Dict, Tuple

class Vertex:
    def __init__(self, idx, pos) -> None:
        self.idx = idx
        self.pos = pos
        self.adj_vs = set()

    def add_adj(self, vid):
        self.adj_vs.add(vid)


class Edge:
    def __init__(self, v1, v2) -> None:
        self.v1 = v1
        self.v2 = v2
        self.adj_fs = set()
    
    def add_adj(self, fid):
        self.adj_fs.add(fid)


class MidVert:
    def __init__(self, idx, v1, v2, pos=None) -> None:
        self.v1 = v1
        self.v2 = v2
        self.idx = idx
        self.pos = pos


class Graph:
    def __init__(self, verts, faces) -> None:
        self.verts_num = verts.shape[0]
        self.face_num = faces.shape[0]

        self.verts: List[Vertex] = [Vertex(i, verts[i]) for i in range(self.verts_num)]
        self.edges: Dict[Tuple[int], Edge] = dict()

        for i in range(self.face_num):
            verts = faces[i]
            verts = np.sort(verts)
            v1, v2, v3 = verts  # v1 < v2 < v3
            
            self.verts[v1].add_adj(v2)
            self.verts[v1].add_adj(v3)
            self.verts[v2].add_adj(v1)
            self.verts[v2].add_adj(v3)
            self.verts[v3].add_adj(v1)
            self.verts[v3].add_adj(v2)

            if (v1, v2) not in self.edges:
                self.edges[(v1, v2)] = Edge(v1, v2)
           
            if (v2, v3) not in self.edges:
                self.edges[(v2, v3)] = Edge(v2, v3)
            
            if (v1, v3) not in self.edges:
                self.edges[(v1, v3)] = Edge(v1, v3)

            self.edges[(v1, v2)].add_adj(i)
            self.edges[(v2, v3)].add_adj(i)
            self.edges[(v1, v3)].add_adj(i)

    def is_interior_edge(self, v1, v2):
        if v1 > v2:
            v1, v2 = v2, v1
        return len(self.edges[(v1, v2)].adj_fs) == 2
    
    def get_opposite_vert(self, fid, v1, v2):
        verts = self.faces[fid]
        for v in verts:
            if v != v1 and v != v2:
                return v
        return -1
    
    def get_all_opposite_verts(self, v1, v2):
        if v1 > v2:
            v1, v2 = v2, v1
        tmp = [self.get_opposite_vert(fid, v1, v2) for fid in self.edges[(v1, v2)].adj_fs]
        assert len(tmp) == 2
        return tmp[0], tmp[1]
    
    def get_all_neighbour_verts(self, v):
        return self.verts[v].adj_vs
    


            
    