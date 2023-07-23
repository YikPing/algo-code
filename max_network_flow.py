from collections import deque

# ==================== Network Flow ====================
class FlowGraph:

    def __init__(self, n, connections, maxIn, maxOut, origin, targets):
        """
        Function description: Create a flow graph from the given inputs
        
        Approach description: Build a graph with 2n + 1 vertices, where n is the number of data centres. Additional intermediate vertex is created to connect to each vertex based 
                              on the maximum(minimum between maxIn and maxOut) they should receive from other vertex. But for origin, their maximum would be the maxOut as we are 
                              only concern with how much they can send out.And for targets, their maximum would be the maxIn as we are only concent with how much they can receive.
                              Then connect all vertices with edges based on the connections given. Then connect all intermediate vertices to their corresponding vertices with 
                              edges based on the maximum(minimum between maxIn and maxOut) they should receive from other vertex.

                              1. Create a graph with 2n + 1 vertices, where n is the number of data centres. Time & aux: O(D)

                              2. Add edges to each vertex based on the connections given. Time: O(C), aux: O(D)

                              3. Add edges from intermediate vertices to their corresponding. Time & aux: O(D)

                              Time Complexity: O(D + C), where D is the number of data centres and C is the number of connections

                              Aux Complexity: O(D), where D is the number of data centres
        :Input:
            n: number of data centres
            connections: list of connections between data centres
            maxIn: list of maximum input for each data centre
            maxOut: list of maximum output for each data centre
            origin: origin data centre
            targets: list of target data centres

        :Output, return or postcondition:
            None 

        :Time complexity: O(D + C), where D is the number of data centres and C is the number of connections

        :Aux space complexity: O(D), where D is the number of data centres

        """
        # create graph with 2n + 1 vertices, O(D)
        # 2n*1 to include intermediate vertices and super target
        self.graph = [None] * ((2*n)+1)
        for i in range(n):
            self.graph[i] = Vertex(i, maxIn[i], maxOut[i])
        #add intermediate vertices
        for i in range(n, 2*n):
            self.graph[i] = Vertex(i, 0, 0)
        #change origin's max, only concern with maximum output of origin
        self.graph[origin].max = maxOut[origin]
        #change target's max, only concern with maximum input of targets
        for target in targets:
            self.graph[target].max = maxIn[target]

        #connect all vertices, O(C)
        for connection in connections:
            u = self.graph[connection[0]]
            #+5 to get intermediate vertices
            v = self.graph[connection[1]+n]
            w = connection[2]
            
            u.add_edges(Edge(u,v,w))  
        
        #add edges from intermediate vertices to corresponding vertices, O(D)
        for i in range(n):
            u = self.graph[i+n]
            v = self.graph[i]
            u.add_edges(Edge(u,v,v.max))

    def connect_targets(self, targets):
        """
        Function description: connect all targets to a super target 

        :Input:
            targets: list of target vertices

        :Output, return or postcondition:
            None

        :Time complexity: O(D), where D is the number of data centres

        :Aux space complexity: O(1)

        """
        n = len(self.graph)
        self.graph[n-1] = Vertex(n-1, float("inf"), 0)
        super_target = self.graph[n-1]

        #connect all targets to super target
        for target in targets:
            target_vertex = self.graph[target]
            target_vertex.add_edges(Edge(target_vertex, super_target, target_vertex.max))

class ResidualGraph:

    def __init__(self, graph, n, origin):
        """
        Function description: Create a residual graph from a flow graph

        :Input:
            graph: network flow graph
            n: number of data centres
            origin: id of origin data centre

        :Output, return or postcondition:
            None

        :Time complexity: O(D + C), where D is the number of data centres and C is the number of connections

        :Aux space complexity: O(D), where D is the number of data centres

        """
        self.residual_graph = [None] * len(graph.graph)
        #add vertex to residual graph, O(D)
        for u in graph.graph:
            self.residual_graph[u.id] = Vertex(u.id, u.max, u.max)
        #add edges to residual graph, O(C)
        for u in graph.graph:
            # residual edge for each edge in the graph
            for edge in u.edges:
                u = self.residual_graph[edge.u.id]
                v = self.residual_graph[edge.v.id]

                residual_edge = Edge(u, v, 0, flow = edge.capacity-edge.flow)

                u.add_edges(residual_edge)

        # origin's intermediate vertex 
        self.origin = self.residual_graph[origin+n]
        # super target at the end of the residual graph
        self.target = self.residual_graph[-1]
        # residual capacity of the path
        self.residual_capacity = None
        # path from origin to target
        self.path = None

    def bfs(self):
        """
        Function description: Breadth first search, starting from origin to target
            
        Approach description: Use a queue to store vertices to be discovered. Pop the first vertex in the queue and mark it as visited.
                              If the target is reached, backtrack to get the path. If not, add all unvisited vertices connected to 
                              the current vertex to the queue while also saving the edge used to reach all the unvisited vertice's
                              previous. 

                              1. All vertices and edges will be explored at most once, time: O(D + C), aux: O(D)

                              2. when the target is reached, backtrack to get the path, time: O(D), aux: O(D)

                              Time Complexity: O(D + C) + O(D)
                                              = O(D + C), where D is the number of data centres and C is the number of connections
        
                              Aux Space Complexity: O(D) + O(D)
                                                  = O(D), where D is the number of data centres 
        :Input:
            None

        :Output, return or postcondition:
            None

        :Time complexity: O(D + C), where D is the number of data centres and C is the number of connections

        :Aux space complexity: O(D),where D is the number of data centres

        """
        # queue of vertices to be discovered
        discovered = deque()
        discovered.append(self.origin)
        while len(discovered) > 0:
            u = discovered.popleft()
            u.visited = True
            # if the target node is reached, backtrack to get path and return True
            if u == self.target:
                self.path = []
                current_edge = u.previous
                while current_edge != None:
                    self.path.append(current_edge)
                    # vertex store the edge used to reach it
                    current_edge = current_edge.u.previous
                return True
            for edge in u.edges:
                v = edge.v
                # if the vertex is not discovered, not visited and has positive flow, add to queue
                if v.discovered == False and v.visited == False and edge.flow > 0:
                    v.discovered = True
                    discovered.append(v)
                    # set the previous vertex to store edge used for backtrack
                    v.previous = edge
        # if the targetis not reachable from origin, return False
        return False
    
    def has_AugmentingPath(self):  
        """
        Function description: Reset all vertices in residual graph and run bfs to find the shortest path from origin to target.

        :Input:
            None

        :Output, return or postcondition:
            None

        :Time complexity: O(D + C), where D is the number of data centres and C is the number of connections

        :Aux space complexity: O(D), where D is the number of data centres

        """
        # reset all vertices to undiscovered and unvisited 
        for vert in self.residual_graph:
            vert.discovered = False
            vert.visited = False
            vert.previous = None
        # run bfs to find the shortest path from origin to super target
        return self.bfs()

    def get_residual_cap(self):
        """
        Function description: Go through all edges in the path and find the residual capacity, which is the minimum flow of all edges in the path

        :Input:
            None

        :Output, return or postcondition:
            None

        :Time complexity: O(D), where D is the number of data centres 

        :Aux space complexity: O(1)

        """
        self.residual_capacity = float("inf")
        for edge in self.path: 
            flow = edge.flow
            if flow < self.residual_capacity:
                self.residual_capacity = flow
        return self.residual_capacity
                

    def augmentFlow(self):
        """
        Function description: Update residual graph by augmenting the flow of the edges in the path

        :Input:
            None

        :Output, return or postcondition:
            None

        :Time complexity: O(D), where D is the number of data centres

        :Aux space complexity: O(1)

        """
        for edge in self.path: 
            # decrease the flow of the reverse edge by the residual capacity
            edge.flow -= self.residual_capacity

            #if reverse edge is not in the graph, add it
            if edge.reverse_edge == None:
                edge.reverse_edge = Edge(edge.v, edge.u, 0, flow = 0)
                edge.v.add_edges(edge.reverse_edge)
            
            # increase the flow of the reverse edge by the residual capacity
            edge.reverse_edge.flow += self.residual_capacity

class Vertex:

    def __init__(self, id, maxIn, maxOut) -> None:
        """
        Function description: Initialize a vertex

        :Input:
            id: id of the vertex
            maxIn: maximum flow into the vertex
            maxOut: maximum flow out of the vertex

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)

        """
        self.id = id
        # max flow vertex should receive
        self.max = min(maxIn, maxOut)
        self.edges = []
        # for bfs
        self.discovered = False
        self.visited = False
        self.previous = None 
    
    def add_edges(self, edge):
        """
        Function description: Add an edge to the vertex

        :Input:
            edge: edge to be added to the vertex

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)

        """
        self.edges.append(edge)

class Edge:

    def __init__(self, u,v,w,flow = 0) -> None:
        """
        Function description: Initialize an edge

        :Input:
            u: vertex u
            v: vertex v
            w: weight of the edge used as capacity
            flow: flow of the edge

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        self.u = u
        self.v = v
        self.flow = flow
        self.capacity = w
        # reverse edge
        self.reverse_edge = None


def ford_fulkerson(my_graph, n, origin):
    """
    ford_fulkerson refered from Malaysia lecture video

    Function description: Find the maximum possible flow from the origin to the super target

    Approach description: create a residual graph from network flow graph. While there is an augmenting path from origin to 
                          super target, find the residual capacity of the path and augment the flow of the edges in the path.

                          1. Create a residual graph from network flow graph, time: O(D + C), aux: O(D)

                          2. While there is an augmenting path from origin to super target, time: O(D + C), aux: O(D)

                                3. Find the residual capacity of the path, time: O(D), aux: O(1)

                                4. Augment the flow of the edges in the path, time: O(D), aux: O(1)
                        
                          Time Complexity: O(D + C) + (O(D + C) * (O(D) + O(D))) 
                                         = O(D + C) + O(D^2 + DC) 
                                         = O(D^2 + DC)
                                         O(DC) can be expressed as O(|D|*|C|^2), since C^2 represents the maximum possible number of 
                                         connections in a graph. Therefore O(|D|*|C|^2) is an upper bound.

                                         = O(|D|*|C|^2), where D is the number of data centres and C is the number of connections

                          Aux Space Complexity: O(D) + (O(D) * (O(1) + O(1)))
                                              = O(D)

    Approach description: 

    :Input:
        my_graph: graph of data centres and connections
        n: number of data centres
        origin: origin data centre

    :Output, return or postcondition:
        None

    :Time complexity: O(|D|*|C|^2), where D is the number of data centres and C is the number of connections

    :Aux space complexity: O(D), where D is the number of data centres 

    """
    #initialize flow to 0
    flow = 0
    #initialize the residual network, O(D + C)
    residual_graph = ResidualGraph(my_graph, n , origin)
    #as long as there is a path from source to sink O(D + C)
    while residual_graph.has_AugmentingPath():
        #get residual capacity of path  O(D)
        residual_cap = residual_graph.get_residual_cap()
        #augment the flow equal to residual capacity 
        flow += residual_cap
        #update the residual network, O(D)
        residual_graph.augmentFlow()
    return flow

# ==================== Q1 Code ====================

def maxThroughput(connections, maxIn, maxOut, origin, targets):
    """
    Function description: Find the maximum possible data throughput from the data centre origin to the data centres in targets
    
    Approach description: Build a flow network graph where each data centre is a vertex and each connection is an edge. Additional intermediate vertex is created to connect
                          to each vertex with edge capacity of the maximum(minimum between maxIn and maxOut) they should receive from other vertex. But for origin, their 
                          maximum would be the maxOut as we are only concern with how much they can send out.And for targets, their maximum would be the maxIn as we are 
                          only concern with how much they can receive. All target vertices are connected to a super target vertex. Then run ford fulkerson to find the max flow.

                          1. Create a flow network graph, Time & Aux: O(D + C)

                          2. Connect all target to super target, Time: O(D), Aux: O(1)

                          3. Run ford fulkerson to get max data throughput, Time: O(|D|*|C|^2), Aux: O(D)
                             - Complexity explained in ford fulkerson function

                          Time Complexity: O(D + C) + O(D) + O(|D|*|C|^2)
                                          =O(|D|*|C|^2)

                          Space Complexity: O(D) + O(1) + O(D)
                                          = O(D)

    :Input:
        connections: a list of tuples (a, b, t) where a ID of the data centre from which the communication channel departs while b is the ID of the data centre 
                     at which the communication channel arrives and t is the maximum throughput of the connection
        maxIn: a list of integers in which maxIn[i] specifies the maximum amount of incoming data that data centre i can process per second
        maxOut: is a list of integers in which maxOut[i] specifies the maximum amount of outgoing data that data centre i can process per second
        origin: ID of the data centre where the data to be backed up is located
        targets: a list ID of targets of data centres

    :Output, return or postcondition:
        max_flow: the maximum possible data throughput from the data centre origin to the data centres specified in targets

    :Time complexity: O(|D|*|C|^2), where D is the number of data centres and C is the number of connections

    :Aux space complexity: O(D), where D is the number of data centres

    """
    num_vertices = len(maxIn) 
    #create a flow network graph, time: O(D + C), aux: O(D + C)
    my_graph = FlowGraph(num_vertices,connections, maxIn, maxOut, origin, targets)
    #connect all target to super target, time: O(D), aux: O(1)
    my_graph.connect_targets(targets)
    #run ford fulkerson to get max flow, time: O(|D|*|C|^2), aux: O(D + C)
    max_throughput = ford_fulkerson(my_graph, num_vertices , origin)
    return max_throughput

"""
Test Cases
"""
connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000), (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
maxIn = [5000, 3000, 3000, 3000, 2000]
maxOut = [5000, 3000, 3000, 2500, 1500]
origin = 0
targets = [4, 2]
expected = 4500
print("Expected:", expected)
print("Actual:  ", maxThroughput(connections, maxIn, maxOut, origin, targets))

connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000), (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
maxIn = [5000, 3000, 3000, 3000, 2000]
maxOut = [5000, 3000, 3000, 2500, 1500]
origin = 0
targets = [2, 3]
expected = 5000
print("Expected:", expected)
print("Actual:  ", maxThroughput(connections, maxIn, maxOut, origin, targets))

connections = [(0, 1, 1000), (0, 2, 2000), (1, 2, 500), (1, 3, 1500), (2, 3, 1000), (2, 4, 2000)]
maxIn = [2000, 3000, 2500, 1500, 2000]
maxOut = [2000, 2500, 3000, 2000, 2500]
origin = 0
targets = [3, 4]
expected = 2000
print("Expected:", expected)
print("Actual:  ", maxThroughput(connections, maxIn, maxOut, origin, targets))

connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000), (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
maxIn = [5000, 3000, 3000, 3000, 2000]
maxOut = [5000, 3000, 3000, 2500, 1500]
origin = 1
targets = [3]
expected = 1000
print("Expected:", expected)
print("Actual:  ", maxThroughput(connections, maxIn, maxOut, origin, targets))