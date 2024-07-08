# Mars Rover Path Planning

This project implements three search algorithms to solve a path planning problem for a Mars rover. The algorithms are designed to find the shortest path from a start location to a goal location on a discretized terrain map of Mars, taking into account energy constraints for uphill travel.

## Algorithms Implemented

1. **Breadth-First Search (BFS):**
   - Simple search algorithm that expands all nodes at the present depth before moving on to nodes at the next depth level.
   
2. **Uniform-Cost Search (UCS):**
   - A variant of Dijkstra's algorithm that considers the 2D Euclidean distance between nodes for path cost calculation.
   
3. **A* Search (A*):**
   - An informed search algorithm that uses both path cost and an admissible heuristic to find the shortest path efficiently. It considers the 3D Euclidean distance.

## Input and Output

- The program reads the input from a file named `input.txt` located in the current directory.
- The output is written to a file named `output.txt` in the same directory.

### Input File Format

1. **First Line:** The name of the algorithm to use (`BFS`, `UCS`, or `A*`).
2. **Second Line:** An integer representing the rover's uphill energy limit.
3. **Third Line:** An integer N representing the number of safe locations.
4. **Next N Lines:** Each line contains a safe location in the format `name x y z`.
5. **Next Line:** An integer M representing the number of safe path segments.
6. **Next M Lines:** Each line contains a safe path segment in the format `nameone nametwo`.

### Output File Format

- A single line containing a space-separated list of location names from the start to the goal.
- If no path is found, the output is `FAIL`.

