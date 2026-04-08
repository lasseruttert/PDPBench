import json
from pathlib import Path
from datetime import datetime

class JSONResultLogger:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def save_result(self, instance_name, size, algorithm, solution, runtime, **kwargs):
        filename = f"{instance_name}_size{size}_{algorithm}.json"
        filepath = self.results_dir / filename
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "instance": instance_name,
            "size": size,
            "algorithm": algorithm,
            "num_vehicles": solution.num_vehicles_used,
            "total_distance": solution.total_distance,
            "runtime_seconds": runtime,
            "is_feasible": solution.is_feasible,
            "routes": solution.routes,
            **kwargs
        }
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)