import heapq
import math

class Node:
    def __init__(self, name, x, y, z):
        self.name = name
        self.x = x
        self.y = y
        self.z = z

def read_input(file_name):
    with open(file_name, 'r') as file:
        algorithm = file.readline().strip()
        uphill_energy_limit = int(file.readline().strip())
        num_locations = int(file.readline().strip())
        locations = {}
        start_node, goal_node = None, None

        for _ in range(num_locations):
            name, x, y, z = file.readline().split()
            node = Node(name, int(x), int(y), int(z))
            locations[name] = node
            if name == 'start':
                start_node = node
            elif name == 'goal':
                goal_node = node

        num_segments = int(file.readline().strip())
        segments = set()

        for _ in range(num_segments):
            name_one, name_two = file.readline().split()
            segments.add((name_one, name_two))
            segments.add((name_two, name_one))

    return algorithm, uphill_energy_limit, start_node, goal_node, locations, segments

def write_output(file_name, path):
    with open(file_name, 'w') as file:
        if path is None:
            file.write("FAIL\n")
        else:
            file.write(" ".join(node for node in path) + "\n")

def calculate_energy(node1, node2):
    return node2.z - node1.z

def bfs_search(start, goal, locations, segments, uphill_energy_limit):
    visited = set()
    queue = [[[start], 0]]  

    while queue:
        path, momentum = queue.pop(0)
        node = path[-1]

        if node == goal:
            return path

        for neighbor in segments[node]:
            if (node, neighbor) not in visited:
                energy_required = calculate_energy(locations[node], locations[neighbor])
                if energy_required <= uphill_energy_limit + momentum:
                    new_path = list(path)
                    new_path.append(neighbor)
                    new_momentum = max(0, - energy_required)
                    queue.append([new_path, new_momentum])
                    visited.add((node, neighbor))
    return None

def ucs_search(start, goal, locations, segments, uphill_energy_limit):
    visited = set()
    queue = [(0, 0, [start])]

    while queue:
        cost, momentum, path = heapq.heappop(queue)
        node = path[-1]

        if node == goal:
            return path

        for neighbor in segments[node]:
            if (node, neighbor) not in visited:
                energy_required = calculate_energy(locations[node], locations[neighbor])
                if energy_required <= uphill_energy_limit + momentum:
                    new_path = list(path)
                    new_path.append(neighbor)
                    new_momentum = max(0, - energy_required)
                    new_cost = cost + math.sqrt((locations[node].x - locations[neighbor].x)**2 + (locations[node].y - locations[neighbor].y)**2)
                    heapq.heappush(queue, (new_cost, new_momentum, new_path))
                    visited.add((node, neighbor))

    return None

def a_star_search(start, goal, locations, segments, uphill_energy_limit):
    visited = set()
    queue = [(0, 0, 0, [start])]

    while queue:
        total_cost, cost, momentum, path = heapq.heappop(queue)
        node = path[-1]

        if node == goal:
            return path

        for neighbor in segments[node]:
            if (node, neighbor) not in visited:
                energy_required = calculate_energy(locations[node], locations[neighbor])
                if energy_required <= uphill_energy_limit + momentum:
                    new_path = list(path)
                    new_path.append(neighbor)
                    new_momentum = max(0, - energy_required)
                    new_cost = cost + math.sqrt((locations[node].x - locations[neighbor].x)**2 + (locations[node].y - locations[neighbor].y)**2 + (locations[node].z - locations[neighbor].z)**2)
                    heuristic = math.sqrt((locations[neighbor].x - locations[goal].x)**2 + (locations[neighbor].y - locations[goal].y)**2 + (locations[neighbor].z - locations[goal].z)**2)
                    total_cost = new_cost + heuristic
                    heapq.heappush(queue, (total_cost, new_cost, new_momentum, new_path))
                    visited.add((node, neighbor))

    return None

def generate_segments(locations, raw_segments):
    segments = {location: set() for location in locations.keys()}

    for segment in raw_segments:
        segments[segment[0]].add(segment[1])

    return segments

def main():
    algorithm, uphill_energy_limit, start, goal, locations, raw_segments = read_input("input.txt")
    segments = generate_segments(locations, raw_segments)

    if algorithm == "BFS":
        path = bfs_search(start.name, goal.name, locations, segments, uphill_energy_limit)
    elif algorithm == "UCS":
        path = ucs_search(start.name, goal.name, locations, segments, uphill_energy_limit)
    elif algorithm == "A*":
        path = a_star_search(start.name, goal.name, locations, segments, uphill_energy_limit)

    write_output("output.txt", path)

if __name__ == "__main__":
    main()
