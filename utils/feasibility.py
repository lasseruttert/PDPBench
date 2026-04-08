from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

def is_feasible(problem: PDPTWProblem, solution: PDPTWSolution, use_prints = False) -> bool:
    """Checks if a given PDPTW solution is feasible with respect to capacity and time windows."""
    pickup_to_delivery = problem.pickup_to_delivery
    delivery_to_pickup = problem.delivery_to_pickup
    
    demands = problem.demands
    vehicle_capacity = problem.vehicle_capacity
    distance_matrix = problem.distance_matrix
    time_windows = problem.time_windows
    service_times = problem.service_times
    nodes = problem.nodes
    
    seen_total = set()
    for route in solution.routes:
        load = 0
        current_time = 0
        seen = set()
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # * Check if the route start at the depot
            if i == 0 and from_node != 0:
                if use_prints: print("Route does not start at depot:", route)
                return False
            
            # * Check if the route ends at the depot
            if i+1 == len(route) - 1 and to_node != 0:
                if use_prints: print("Route does not end at depot:", route)
                return False
            
            # * Check if depot is visited in the middle of the route
            if i+1 < len(route) - 1 and to_node == 0:
                if use_prints: print("Depot visited in the middle of route:", route)
                return False  
            
            # * Check if node has already been served
            if to_node in seen:
                if use_prints: print("Node visited multiple times:", to_node)
                return False 
             
            # * Check if pickup happens before delivery
            if to_node in delivery_to_pickup:
                pickup = delivery_to_pickup[to_node]
                if pickup not in seen:
                    if use_prints: print("Delivery before pickup:", to_node)
                    return False  
            # * Check if node is valid (depot, pickup, or delivery)
            else:
                if to_node not in pickup_to_delivery and to_node != 0:
                    if use_prints: print("Invalid node:", to_node)
                    return False  
                
            # * Check if vehicle capacities are respected
            load += demands[to_node]
            if load < 0 or load > vehicle_capacity:
                if use_prints: print("Capacity violation at node:", to_node)
                return False
            
            # * Check if time windows are respected
            travel_time = distance_matrix[from_node, to_node]
            current_time += travel_time
            
            tw_start, tw_end = time_windows[to_node]
            if current_time < tw_start:
                current_time = tw_start
            if current_time > tw_end:
                if use_prints: print("Arrived too late at node:", to_node)
                return False 
            current_time += service_times[to_node]
            
            seen.add(to_node)
        
        seen_total.update(seen)

    # * Check if all nodes are served        
    if seen_total != set(node.index for node in nodes):
        not_served = set(node.index for node in nodes) - seen_total
        if use_prints: print("Not all nodes served. Not served:", not_served)
        return False  
    
    return True
