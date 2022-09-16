from typing import List


class Vertex:
    def __init__(self, node_id: int):
        self.id = node_id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return str(self.id)

    def uid(self):
        return self.id

    def __repr__(self):
        return str(self)


class Edge:
    def __init__(self, src: Vertex, dst: Vertex, weight: float = 0):
        self.src = src
        self.dst = dst
        self.weight = weight

    def uid(self):
        return "{}_{}".format(self.src.uid(), self.dst.uid())

    def __str__(self):
        return self.uid()

    def __repr__(self):
        return str(self)


class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        # self.vert_dict = {}
        self.num_vertices = 0

    def add_vertex(self, vertex: Vertex):
        self.num_vertices = self.num_vertices + 1
        self.vertices[vertex.uid()] = vertex
        return vertex

    def get_vertex(self, node_uid: int) -> Vertex:
        return self.vertices[node_uid]

    def get_vertices(self) -> List[Vertex]:
        return list(self.vertices.values())

    def get_edge(self, edge_uid: str) -> Edge:
        return self.edges[edge_uid]

    def find_edges(self, vertex: Vertex) -> List[Edge]:
        """Find directional edges going out from vertex"""
        return self.edges[vertex.uid()]

    def add_edge(self, frm: Vertex, to: Vertex, cost=0, directed: bool = False):
        if frm.uid() not in self.vertices:
            self.add_vertex(frm)
        if to.uid() not in self.vertices:
            self.add_vertex(to)

        edge = Edge(frm, to, cost)
        if frm.uid() not in self.edges:
            self.edges[frm.uid()] = [edge]
        else:
            self.edges[frm.uid()].append(edge)
        if not directed:
            edge = Edge(to, frm, cost)
            if to.uid() not in self.edges:
                self.edges[to.uid()] = [edge]
            else:
                self.edges[to.uid()].append(edge)

    def traverse(self, starting_node_id: int = None):
        if starting_node_id is None:
            starting_node_id = list(self.vertices.keys())[0]
        starting_node = self.get_vertex(starting_node_id)
        return dfs(self, starting_node, [], [])


def dfs(graph: Graph, node: Vertex, visited: List[Vertex], directions):
    if node not in visited:
        visited.append(node)
        for edge in graph.find_edges(node):
            # n = edge.src if edge.dst == node.uid() else edge.dst
            n = edge.dst
            directions.append(edge)
            dfs(graph, n, visited, directions)
    return directions
