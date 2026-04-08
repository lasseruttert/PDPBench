from .pdptw_problem import PDPTWProblem, Node
import numpy as np

def mendeley_reader(file_path):
    """Reads a PDPTW problem instance from a Mendeley/Sartori & Buriol formatted file.

    These instances are based on real-world data with lat/long coordinates and
    pre-computed travel times from OSRM road network data.

    Returns:
        PDPTWProblem: The parsed PDPTW problem instance.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    metadata = {}
    line_idx = 0

    while line_idx < len(lines):
        line = lines[line_idx].strip()

        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            metadata[key] = value

        if line == 'NODES':
            line_idx += 1
            break

        line_idx += 1

    size = int(metadata.get('SIZE', 0))
    capacity = int(metadata.get('CAPACITY', 0))

    nodes = []
    for i in range(size):
        parts = lines[line_idx].strip().split()
        node = Node(
            index=int(parts[0]),
            x=float(parts[1]),  
            y=float(parts[2]),  
            demand=int(parts[3]),
            time_window=(int(parts[4]), int(parts[5])),
            service_time=int(parts[6]),
            pickup_index=int(parts[7]),
            delivery_index=int(parts[8])
        )
        nodes.append(node)
        line_idx += 1

    while line_idx < len(lines) and lines[line_idx].strip() != 'EDGES':
        line_idx += 1

    line_idx += 1  
    distance_matrix = np.zeros((size, size))
    for i in range(size):
        if line_idx >= len(lines):
            break
        row_values = list(map(int, lines[line_idx].strip().split()))
        distance_matrix[i, :] = row_values
        line_idx += 1

    total_demand = sum(node.demand for node in nodes if node.demand > 0)
    num_vehicles = max(1, (total_demand + capacity - 1) // capacity)

    import os
    problem_name = os.path.splitext(os.path.basename(file_path))[0]

    problem = PDPTWProblem(
        num_vehicles=num_vehicles,
        vehicle_capacity=capacity,
        nodes=nodes,
        distance_matrix=distance_matrix,
        name=problem_name,
        dataset="Mendeley"
    )

    return problem

if __name__ == "__main__":
    problem = mendeley_reader('G:/Meine Ablage/rl-memetic-pdptw/data/n100/bar-n100-1.txt')
    print(problem)
