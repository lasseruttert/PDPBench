from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

import os

class BestKnownSolutions:
    def __init__(self, bks_path: str = "bks"):
        self.bks_path = bks_path

    def get_bks_as_tuple(self, problem: PDPTWProblem) -> tuple:
        problem_name = problem.name.lower()
        dataset = problem.dataset
        subfolder = ""
        if dataset == "Li & Lim":
            subfolder = "li_lim"
        elif dataset == "Mendeley":
            subfolder = "mendeley"
        
        folder_path = os.path.join(self.bks_path, subfolder)
        for file_name in os.listdir(folder_path):
            if file_name.startswith(problem_name + "."):
                parts = file_name[len(problem_name) + 1:-4].split("_")
                num_vehicles = int(parts[0])
                total_distance = float(parts[1])
                return (num_vehicles, total_distance)

    
    def get_bks_as_solution(self, problem: PDPTWProblem) -> PDPTWSolution:
        problem_name = problem.name.lower()
        dataset = problem.dataset
        subfolder = ""
        if dataset == "Li & Lim":
            subfolder = "li_lim"
        elif dataset == "Mendeley":
            subfolder = "mendeley"
        
        folder_path = os.path.join(self.bks_path, subfolder)
        # the file we want has lines called Route X: ...
        for file_name in os.listdir(folder_path):
            if file_name.startswith(problem_name + "."):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                
                routes = []
                for line in lines:
                    if line.startswith("Route"):
                        parts = line.split(":")
                        route_str = parts[1].strip()
                        if route_str:
                            route = [int(node) for node in route_str.split()]
                        else:
                            route = []
                        routes.append(route)
                        
                for route in routes:
                    if route[0] != 0:
                        route.insert(0, 0)
                    if route[-1] != 0:
                        route.append(0)
                
                solution = PDPTWSolution(problem=problem, routes=routes)
                return solution
        
        pass