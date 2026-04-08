from dataclasses import dataclass
import numpy as np
from typing import Optional
from utils.pdptw_problem import PDPTWProblem
from copy import deepcopy
import re

# Try to import C++ solution module, fall back to Python if not available
try:
    from utils import solution_core
    _USE_CPP_SOLUTION = True
except ImportError:
    _USE_CPP_SOLUTION = False
@dataclass
class PDPTWSolution:
    problem: PDPTWProblem
    """A solution to a PDPTW problem instance."""
    routes: list[list[int]]
    """A list of routes, where each route is a list of node indices starting and ending with the depot (index 0)."""
    
    def __post_init__(self):
        self._total_distance = None
        self._route_lengths = None
        self._is_feasible = None
        self._encoding = None
        self._hashed_encoding = None
        self._node_to_route = None
        
    def _clear_cache(self):
        self._total_distance = None
        self._route_lengths = None
        self._is_feasible = None
        self._encoding = None
        self._hashed_encoding = None
        self._node_to_route = None

    def __len__(self):
        return len(self.routes)
    
    def __iter__(self):
        return iter(self.routes)

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the solution. """
        return {
            'routes': self.routes,
            'total_distance': self.total_distance,
            'num_vehicles_used': self.num_vehicles_used,
            'num_customers_served': self.num_customers_served,
            'encoding': self.encoding,
            'hashed_encoding': self.hashed_encoding
        }
        
    def __str__(self) -> str:
        header = f"\033[1;33mPDPTW Solution\033[1;30m | \033[1;33mVehicles used: {self.num_vehicles_used}\033[1;30m | \033[1;33mDistance: {self.total_distance:.2f}\033[0m"
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        visible_text = ansi_escape.sub('', header)
        
        line = "\033[1;35m" + "=" * len(visible_text) + "\033[0m"
        route_lines = []
        for i, route in enumerate(self.routes):
            if len(route) > 2:
                route_lines.append(f"  \033[1;36mVehicle {i+1}:\033[0m {' - '.join(map(str, route))}")
        return (
        line + "\n" +
        header + "\n" +
        line + "\n" +
        "\n".join(route_lines) + "\n"
    )
    
    @property
    def total_distance(self) -> float:
        """Calculates and returns the total distance of all routes in the solution."""
        if self._total_distance is None:
            if _USE_CPP_SOLUTION:
                total_distance, route_lengths = solution_core.calculate_total_distance(
                    self.routes,
                    self.problem.distance_matrix
                )
                self._total_distance = total_distance
                self._route_lengths = route_lengths
            else:
                # Fallback to Python/NumPy implementation
                distance_matrix = self.problem.distance_matrix
                total_distance = 0.0
                route_lengths = {}

                for idx, route in enumerate(self.routes):
                    if len(route) < 2:
                        route_lengths[idx] = 0.0
                        continue

                    from_nodes = np.array(route[:-1], dtype=np.int32)
                    to_nodes = np.array(route[1:], dtype=np.int32)
                    distances = distance_matrix[from_nodes, to_nodes]
                    route_distance = distances.sum()

                    route_lengths[idx] = route_distance
                    total_distance += route_distance

                self._total_distance = total_distance
                self._route_lengths = route_lengths
        return self._total_distance
    
    @property
    def route_lengths(self) -> np.ndarray:
        """Returns a dictionary mapping each route index to its total distance."""
        if self._route_lengths is None:
            self._route_lengths = {}
            _ = self.total_distance 
        return self._route_lengths
    
    @property
    def is_feasible(self) -> bool:
        """Checks and returns whether the solution is feasible."""
        if self._is_feasible is None:
            from utils.feasibility import is_feasible
            self._is_feasible = is_feasible(self.problem, self)
        return self._is_feasible
    
    @property
    def encoding(self) -> str:
        """Returns a string encoding of the solution."""
        if self._encoding is None:
            # convert routes to a string, sort the strings based on length and then concatenate
            route_strings = ['-'.join(map(str, route)) for route in self.routes if len(route) > 2]
            route_strings.sort(key=lambda x: len(x))
            self._encoding = '|'.join(route_strings)
        return self._encoding
    
    @property
    def hashed_encoding(self) -> int:
        """Returns a hash of the solution's encoding."""
        if self._hashed_encoding is None:
            self._hashed_encoding = hash(self.encoding)
        return self._hashed_encoding
    
    @property
    def node_to_route(self) -> dict[int, int]:
        """Returns a mapping from node indices to their corresponding route index."""
        if self._node_to_route is None:
            if _USE_CPP_SOLUTION:
                pickup_nodes_list = list(self.problem.pickup_nodes)
                delivery_nodes_list = list(self.problem.delivery_nodes)
                self._node_to_route = solution_core.build_node_to_route(
                    self.routes,
                    pickup_nodes_list,
                    delivery_nodes_list
                )
            else:
                # Fallback to Python implementation
                mapping = {}
                pickup_nodes = self.problem.pickup_nodes
                delivery_nodes = self.problem.delivery_nodes
                for route_idx, route in enumerate(self.routes):
                    for node in route:
                        if node in pickup_nodes or node in delivery_nodes:
                            mapping[node] = route_idx
                self._node_to_route = mapping
        return self._node_to_route
    
    @property
    def num_vehicles_used(self) -> int:
        """Returns the number of vehicles used in the solution (i.e., routes with more than just depot nodes)."""
        return sum(1 for r in self.routes if len(r) > 2)
    
    @property
    def num_customers_served(self) -> int:
        """Returns the number of customers served in the solution."""
        return sum(len(r) - 2 for r in self.routes if len(r) > 2)
    
    @property
    def visited_nodes(self) -> list[int]:
        """Returns a list of all visited nodes in the solution, excluding depot nodes."""
        visited = []
        for route in self.routes:
            visited.extend([node for node in route if node != 0])
        return visited
    
    def modify_routes(self, new_routes: list[list[int]]):
        """Modifies the routes of the solution and clears cached properties."""
        self.routes = new_routes
        self._clear_cache()
        
    def check_feasibility(self) -> bool:
        """Re-evaluates and returns the feasibility of the solution."""
        from utils.feasibility import is_feasible
        self._is_feasible = is_feasible(self.problem, self, use_prints=True)
        return self._is_feasible
    
    def get_served_requests(self, problem) -> list[int]:
        """Returns a list of served customer requests based on the problem instance."""
        served = set()
        seen = set()
        for route in self.routes:
            for node in route:
                if problem.is_pickup(node):
                    seen.add(node)
                elif problem.is_delivery(node):
                    pair = problem.get_pair(node)
                    if pair[0] in seen:
                        served.add(pair)
        return list(served)
    
    def get_unserved_requests(self, problem) -> list[int]:
        """Returns a list of unserved customer requests based on the problem instance."""
        served = set()
        seen = set()
        unserved = []
        for route in self.routes:
            for node in route:
                if problem.is_pickup(node):
                    seen.add(node)
                elif problem.is_delivery(node):
                    pair = problem.get_pair(node)
                    if pair[0] in seen:
                        served.add(pair)
        for request in problem.pickups_deliveries:
            if request not in served:
                unserved.append(request)
        return unserved
    
    def remove_request(self, problem: PDPTWProblem, request: int):
        """Removes a specific customer request (both pickup and delivery) from the solution."""
        pickup, delivery = problem.get_pair(request)
        route = self.node_to_route.get(pickup)
        if route is not None:
            self.routes[route] = [node for node in self.routes[route] if node != pickup and node != delivery]
        self._clear_cache()
    
    def clone(self) -> 'PDPTWSolution':
        """Creates and returns a deep copy of the solution."""
        routes = [route[:] for route in self.routes]
        new_solution = PDPTWSolution(self.problem, routes=routes)

        return new_solution
    
    def get_solution_txt(self) -> str:
        """Generates a textual representation of the solution in a specific format."""
        lines = []
        lines.append(f"Instance name : {self.problem.name}")
        lines.append(f"Authors       : Placeholder")
        lines.append(f"Date          : Placeholder")
        lines.append(f"Reference     : Placeholder")
        lines.append(f"Solution")
        # Vehicle routes - Format: Route # : node1 node2 ... nodeN (no depot at start/end)
        index = 1
        for i, route in enumerate(self.routes):
            if len(route) > 2:
                route_no_depot = route[1:-1]  # Exclude starting and ending depot
                route_str = ' '.join(map(str, route_no_depot))
                lines.append(f"Route {index} : {route_str}")
                index += 1  
        return '\n'.join(lines)

    def save_solution_txt(self, save_path: str):
        """Saves the textual representation of the solution to a file."""
        solution_txt = self.get_solution_txt()
        # make sure save_path exists
        import os   
        os.makedirs(save_path, exist_ok=True)
        file_path = f"{save_path}/{self.problem.name}.{self.num_vehicles_used}_{int(self.total_distance)}.txt"
        # Force Unix line endings with newline='\n' and UTF-8 encoding
        with open(file_path, 'w', newline='\n', encoding='utf-8') as file:
            file.write(solution_txt)