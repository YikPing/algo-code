"""
Min Heap with fixed heap implemented using list, with a index list mapping for O(1) update of vertex distance
"""
class MinHeap():
    def __init__(self, vertices) -> None:
        """
        Function description: constructs all the necessary attributes for Minheap, all vertices are added to heap as tuple (distance, vertex)

        Attributes:
            length: length of heap
            the_array: Heap list/array, fixed size
            index: fixed sized list index mapping for heap

        :Input:
            vertices: list of vertices

        :Output, return or postcondition:
            None

        :Time complexity: O(N), where N is number of vertex in vertices list

        :Aux space complexity: O(N), where N is number of vertex in vertices list
        """
        n = len(vertices)
        self.length = 0
        self.the_array = [None] * (n + 1)   # O(N)
        self.index = [None] * (n)   # O(N)
        # add vertices to heap, O(N)
        for i in range(n):  
            self.add(vertices[i].distance, vertices[i])

    def __len__(self) -> int:
        """
        Function description: return length of heap

        :Input:
            None

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        return self.length

    def add(self, distance, vertex) -> bool:
        """
        Function description: add element to fixed array/list and rise it to its correct position

        :Input:
            distance: distance of vertex
            vertex: vertex to be added

        :Output, return or postcondition:
            None

        :Time complexity: O(log v), where v is number of element in heap

        :Aux space complexity: O(1)
        """
        self.length += 1
        self.the_array[self.length] = (distance, vertex)
        self.rise(self.length)

    def update(self, v) -> None:
        """
        Function description: update distance of a vertex and rise it to its correct position

        :Input:
            v: vertex to be updated

        :Output, return or postcondition:
            None

        :Time complexity: O(log v), where v is number of element in heap

        :Aux space complexity: O(1)
        """
        # update distance of vertex using index list to find index of vertex in heap
        self.the_array[self.index[v.id]] = (v.distance, v)
        self.rise(self.index[v.id])

    def rise(self, k) -> None:
        """
        Function description: Rise element at index k to its correct position

        :Input:
            k: index of element to be risen

        :Output, return or postcondition:
            None

        :Time complexity: O(log n), where n is number if element in heap

        :Aux space complexity: O(1)
        """
        item = self.the_array[k]
        while k > 1 and item[0] < self.the_array[k // 2][0]:   
            # update index list to reflect new index of vertex in heap, current vertex is at index k
            self.index[self.the_array[k // 2][1].id] = k
            # move current vertex to index of its parent vertex in heap array
            self.the_array[k] = self.the_array[k // 2]
            k = k // 2
        # update index list to reflect new index of item in heap
        self.the_array[k] = item
        # move item to k in heap array
        self.index[item[1].id] = k
    
    def smallest_child(self, k) -> int:
        """
        Function description: Returns the index of k's child with smallest value

        :Input:
            k: index of element to be checked

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        if 2 * k == self.length or \
                self.the_array[2 * k][0] < self.the_array[2 * k + 1][0]:
            return 2 * k
        else:
            return 2 * k + 1

    def sink(self, k) -> None:
        """
        Function description: Make the element at index k sink to the correct position.

        :Input:
            k: index of element to be sunk

        :Output, return or postcondition:
            None

        :Time complexity: O(log n), where n is number if element in heap

        :Aux space complexity: O(1)
        """
        item = self.the_array[k]
        while 2 * k <= self.length:
            # find smallest child of current vertex
            min_child = self.smallest_child(k)
            # if item is smaller than smallest child, break
            if self.the_array[min_child][0] >= item[0]:
                break
            # update index list to reflect new index of the smallest child vertex in heap, current vertex is at index k
            self.index[self.the_array[min_child][1].id] = k
            # move smallest child vertex to index of its parent vertex in heap array
            self.the_array[k] = self.the_array[min_child]
            k = min_child
        # update index list to reflect new index of item in heap
        self.the_array[k] = item
        # move item to k in heap array
        self.index[item[1].id] = k

        
    def serve(self) -> tuple:
        """
        Function description: Remove (and return) the minimum element from the heap.

        :Input:
            None

        :Output, return or postcondition:
            min_elt: minimum element in heap

        :Time complexity: O(log n), where n is number if element in heap

        :Aux space complexity: O(1)
        """
        if self.length == 0:
            raise IndexError
        min_elt = self.the_array[1]
        self.length -= 1
        if self.length > 0:
            # update index list to reflect new index of vertex in heap
            self.index[self.the_array[self.length+1][1].id] = 1     # last vertex in heap is now at index 1
            self.index[self.the_array[1][1].id] = self.length+1     # minimum element in heap now at last index
            # move last vertex in heap to index 1
            self.the_array[1] = self.the_array[self.length+1]
            # move minimum element in heap to last index
            self.the_array[self.length+1] = min_elt

            # sink the vertex at index 1 to its correct position
            self.sink(1)
        # return minimum vertex in heap, at index 1 since tuple is (distance, vertex)
        return min_elt[1]

"""
Graph, Vertex and Edge class 
"""
class Graph:
    def __init__(self, v):
        """
        Function description: Initialize a 2 layer graph with 2 * v number of vertices, 2nd layer vertices start from v and end at (v * 2) - 1 
                              (example: layer 1 has vertices 0-9, layer 2 has vertices 10-19 and 10-19 will represent duplicate of 0-9)

        Attributes:
            vertices: list of vertices in the graph
            have_passenger: boolean to indicate whether there is a passenger

        :Input:
            v : an integer number of vertices

        :Output, return or postcondition: 
            None

        :Time complexity: O(v), where v is number of vertices

        :Aux space complexity: O(v), where v is number of vertices
        """
        # list of vertices with 2 * v number of vertices for 2nd layer, O(v)
        self.vertices = [None] * (v*2)
        # add 1st layer vertices, O(v)
        for i in range(v):
            self.vertices[i] = Vertex(i)

        # 2nd layer id start from the end of 1st layer, O(v)
        for j in range(v, 2*v):
            self.vertices[j] = Vertex(j)

        # indicate whether there is a passenger, initialize as true 
        self.have_passenger = True

    def add_edges(self, argv_edges):
        """
        Function description: Add edges to vertices in the graph, 2 edges created each time for each layer's edge

        :Input:
            argv_edges : list of tuples containing 4 elements, u as starting vertex, v as ending vertex, w_alone as weight of 1st layer edge, w_carpool as weight of 2nd layer edge

        :Output, return or postcondition:
            None

        :Time complexity: O(E), where E is number of edges in the graph 

        :Aux space complexity: O(1)
        """
        for edge in argv_edges: 
            # add 1st layer edge
            u = self.vertices[edge[0]]
            v = self.vertices[edge[1]]
            w_alone = edge[2]
            current_edge = Edge(u, v, w_alone)
            u.add_edge(current_edge)
            
            # add 2nd layer edge
            u_2 = self.vertices[edge[0] + len(self.vertices)//2]
            v_2 = self.vertices[edge[1] + len(self.vertices)//2]
            w_carpool = edge[3]
            current_edge_2 = Edge(u_2, v_2, w_carpool)
            u_2.add_edge(current_edge_2)

    def connect(self, passenger):
        """
        Function description: Add an edge connecting the passenger vertex between both the layers with the weight of 0 (meaning no cost to travel between layers)

        :Input:
            passenger : an integer representing the vertex id where there is a passenger

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        # connect passenger vertex in 1st layer to passenger vertex in 2nd layer (vertex id + len(vertices)//2)
        new_edge = Edge(self.vertices[passenger], self.vertices[passenger + len(self.vertices)//2], 0)
        # add edge to 1st layer passenger vertex
        self.vertices[passenger].add_edge(new_edge)
    
    def dijkstra (self, source, destination): 
        """
        Function description: dijsktra's algorithm for the 2 layer graph, return the shortest path from source to destination.

        Approach description: When there is no passenger, we can terminate the algorithm after reaching destination in 1st layer because there is no need to travel to 2nd layer.
                              When there is a passenger, we terminate the algorithm after reaching/visitng both destination in 1st and 2nd layer. Then the destination with lower distance 
                              is the shortest path and will be used to backtrack to get shortest path.

                              1. Initialize min fixed heap with all vertices, time: complexity O(V), aux space: O(V)

                              2. While loop to iterate through all vertices in heap, time: complexity O(V), aux space: O(1)

                                    3. serving from heap, time: O(log V), aux space: O(1)
                                    4  backtracking to return shortest path, time: O(V), aux space: O(1)
                                    5. relax edges, time:  O(V), aux space: O(1)
                                        6. update heap, time:  O(log V), aux space: O(1)

                            Time complexity: O(V) + ( O(V) * (O(log V) + O(V) + O(V log V)) )
                                            = O(V^2 log V)
                                            ~ O(E log V), where E is number of edges in the graph and V is number of vertices in the graph
                                            can be simplified to O(E log V) because E is bounded by V^2 and greater than V
                                        
                            Aux space complexity: O(V) + ( O(1) * (O(1) + O(1) + O(1)) )
                                                = O(V)

        :Input:
            source : source vertex in integer
            destination : destination vertex in integer

        :Output, return or postcondition:
            path : list of vertices for the shortest path taken to reach destination from source

        :Time complexity: O(E log v), where E is number of edges in the graph and v is number of vertices in the graph

        :Aux space complexity: O(V), where V is number of vertices in graph
        """
        # initialize source vertex
        source.distance = 0
        source.added_to_queue()
        # initialize a discovered min fixed heap, with all vertices, O(V)
        discovered = MinHeap(self.vertices) 
        # initialize flag to indicate if we are reached both layer's destination
        dest1_reached = False
        # if there are passenger, should terminate only after reaching destination in 2nd layer, else terminate after reaching destination in 1st layer
        dest2_reached = not self.have_passenger 
        destination_2 = self.vertices[destination.id + len(self.vertices)//2]   #2nd layer destination vertex = 1st layer destination vertex id + V/2
        while len(discovered) > 0:   
            u = discovered.serve()  # serve #O(logV)
            u.visited = True        # visited u
            # if first layer destination reached, set flag
            if u == destination:
                dest1_reached = True
            # if second layer destination reached, set flag
            elif u == destination_2:
                dest2_reached= True
            if dest1_reached and dest2_reached:
                # if 2nd layer destination shorter, backtrack from 2nd layer destination
                if destination_2.distance < destination.distance:
                    destination = destination_2
                #backtrack to get path, O(V)
                path = []
                cur_vertex = destination
                prev = None
                while cur_vertex != source:
                    # if id is from 2nd layer, convert back to 1st layer id for path
                    if cur_vertex.id >= len(self.vertices)//2:
                        id = cur_vertex.id - len(self.vertices)//2
                    else:
                        id = cur_vertex.id
                    # if id is same as previous id, means this is the edge that connect same vertex of each layer, skip it
                    if id != prev:
                        path.append(id)
                    cur_vertex = cur_vertex.previous
                    # prev used for next iteration, to check if id is same as previous id, it is same when travelling between layers
                    prev = id
                path.append(source.id)  #append source
                path.reverse()  #reverse path to get correct order, O(V)
                return path
            # perform edge relaxation on all adjacent vertices
            for edge in u.edges:    #O(V)
                v = edge.v
                # if not yet discovered, update in heap
                if v.discovered == False:      #means distance is still infinity
                    v.added_to_queue()        #discrovered v, adding to queue
                    v.distance = u.distance + edge.w
                    v.previous = u
                    discovered.update(v)    # update heap O(log V)
                # it is in heap, but not yet finalize
                elif v.visited == False: 
                    # if shorter route is found, update it in heap
                    if v.distance > u.distance + edge.w:
                        # update distance
                        v.distance = u.distance + edge.w
                        v.previous = u
                        # update heap O(log V)
                        discovered.update(v) # update vertex v in heap, with distance (smaller); perform upheap

class Vertex:
    def __init__(self, id):
        """
        Function description: constructor for Vertex class

        Attributes: 
            id: id of vertex
            edges: list of edges
            discovered: flag to indicate if vertex is discovered
            visited: flag to indicate if vertex is visited
            distance: distance from source vertex
            previous: previous vertex in the shortest path

        :Input:
            id: id of vertex

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        # vertex id
        self.id = id
        # list of edges
        self.edges = []
        # for traversals
        self.discovered = False
        self.visited = False
        # distance
        self.distance = float("inf")
        # for backtracking
        self.previous = None
    
    def add_edge(self, edge):
        """
        Function description: add an edge to the vertex

        :Input:
            edge: edge to be added

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        self.edges.append(edge)

    def added_to_queue(self):
        """
        Function description: set discovered to True

        :Input:
            None

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        self.discovered = True

    def visited_node(self):
        """
        Function description: set visited to True

        :Input:
            None

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        self.visited = True

class Edge:
    def __init__(self, u, v, w):
        """
        Function description: constructor for Edge class, u start vertex, v end vertex, w weight

        Attributes:
            u: start vertex
            v: end vertex
            w: weight of edge

        :Input:
            None

        :Output, return or postcondition:
            None

        :Time complexity: O(1)

        :Aux space complexity: O(1)
        """
        self.u = u
        self.v = v
        self.w = w

def optimalRoute(start, end, passengers, roads):
    """
    Function description: This function returns one optimal route to go from start to end with the minimum possible total travel time.

    Approach description: Create a graph with double the amount of vertices to represent 2 layers graph, layer 1 would be the graph with no passenger weight, 
                          layer 2 would be the identical graph but with passenger weight. The 2 graph is connected with vertex where there are potential 
                          passengers with the weight of 0 because there is no time needed to pick up passenger. Then, dijkstra can be perform on the graph 
                          and only terminate when distance of both destination at layer 1 and layer 2 is finalized/reached or terminate when destination at layer 1 
                          is reached when there is no potential passenger since there is no need to explore/go through 2nd layer graph with passenger. 
                          If both destination is reached, then the destination with lower distance will be used to backtrack and return the path. 

                          Given start, end, list of passengers and list of roads:

                          1. find max vertex at list of roads to create graph, time: O(R), aux: O(1)

                          2. Create a 2 layer graph, time: O(L), aux: O(L)

                          3. add all the edges to both layer of graph, time: O(R), aux: O(R)

                          4. connect all the potential passenger vertex between layer 1 and layer 2, time: O(P), aux: O(P) (The locations specified by start and end will not have potential passengers.)
                             P at max is L - 2 (since start and end is not included) therefore, O(P) < O(L)
                          
                          5. perform dijkstra on the graph, time: O(R Log L), aux: O(R + L)

                          Time complexity: O(R) * O(L) + O(R) + O(P) + O(R Log L) 
                                            = O(R Log L), where R is number of roads, L is key locations/vertices

                          Aux space complexity: O(L) + O(R) + O(P) + O(L)
                                            = O(R + L), where R is number of roads, L is key locations/vertices

    :Input:
        start: integer representing the starting vertex
        end: integer representing the ending vertex
        passengers:  List of integer representing the locations where there are potential passengers
        roads: List of tuple representing the roads 

    :Output, return or postcondition: optimal route as a list of integers

    :Time complexity: O(R Log L), where R is number of roads, L is key locations/vertices

    :Aux space complexity: O(R + L), where R is number of roads, L is key locations/vertices
    """
    #find max vertex to create graph, O(R)
    max_vertex = 0
    for u, v, w, p in roads:
        if u > max_vertex:
            max_vertex = u
        if v > max_vertex:
            max_vertex = v

    # Create a 2 layer graph, first graph with alone weight, second graph with passenger weight, time: O(L), aux: O(L)
    g = Graph(max_vertex + 1)   # max_vertex + 1 because vertex start from 0

    # add all edges, layer 1 for alone and layer 2 with passenger, time: O(R), aux: O(R)
    g.add_edges(roads)

    # if there are no passenger, set have_passenger to False so that dijkstra will not explore the 2nd layer (graph with passenger weight)
    if len(passengers) == 0:
        g.have_passenger = False
    # connect edges with passengers between 2 layer, time: O(P), aux: O(P)
    for each_passenger in passengers:
        g.connect(each_passenger)

    # run dijkstra on the 2 layer graph, time: O(R Log L), aux: O(L)
    path = g.dijkstra(g.vertices[start], g.vertices[end])
    return path


"""
Test cases
"""
start = 0
end = 4
passengers = [2, 1]
roads = [
    (0,4,30,5),
    (0,1,5,4),
    (1,3,3,2),
    (3,2,2,1),
    (2,0,1,1)]
result = [0, 1, 3, 2, 0, 4]
print("Expected:", result)
print("Actual:  ",optimalRoute(start, end, passengers, roads))

start = 1
end = 2
passengers = []
roads = [
        (3, 4, 24, 10), 
        (4, 1, 16, 6), 
        (0, 2, 28, 14), 
        (1, 3, 27, 12), 
        (4, 0, 5, 4), 
        (2, 4, 15, 9)]
result = [1, 3, 4, 0, 2]
print("Expected:", result)
print("Actual:  ",optimalRoute(start, end, passengers, roads))

start = 4
end = 9
passengers = [2, 6, 0]
roads = [
        (4, 6, 30, 18), 
        (3, 1, 8, 1), 
        (9, 1, 9, 5), 
        (1, 9, 30, 2), 
        (8, 5, 12, 12),
        (8, 9, 8, 6), 
        (1, 8, 25, 2), 
        (2, 4, 4, 2), 
        (6, 0, 25, 5), 
        (4, 3, 6, 6), 
        (1, 2, 15, 7)
        ]
result = [4, 3, 1, 2, 4, 3, 1, 9] 
print("Expected:", result)
print("Actual:  ",optimalRoute(start, end, passengers, roads))