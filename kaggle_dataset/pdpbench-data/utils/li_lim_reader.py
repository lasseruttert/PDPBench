from pathlib import Path
from .pdptw_problem import PDPTWProblem, Node

def li_lim_reader(file_path):
    """Reads a PDPTW problem instance from a Li & Lim formatted file.

    Returns:
        PDPTWProblem: The parsed PDPTW problem instance.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_vehicles, vehicle_capacity, _ = map(int, lines[0].split())

    nodes = []
    for line in lines[1:]:
        parts = list(map(int, line.strip().split()))
        node = Node(
            index=parts[0],
            x=parts[1],
            y=parts[2],
            demand=parts[3],
            time_window=(parts[4], parts[5]),
            service_time=parts[6],
            pickup_index=parts[7],
            delivery_index=parts[8]
        )
        nodes.append(node)

    problem_name = Path(file_path).stem

    problem = PDPTWProblem(
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        nodes=nodes,
        name=problem_name,
        dataset="Li & Lim"
    )

    return problem

if __name__ == "__main__":
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc101.txt')
    data = problem.data
    problem.num_vehicles
    # print(data)