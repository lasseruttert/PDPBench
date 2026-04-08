from dataclasses import dataclass
import numpy as np
import re

from pyparsing import line

@dataclass
class Node:
    """A node in the PDPTW problem, representing a depot, pickup, or delivery location."""
    index: int 
    """A unique identifier for the node. Used to reference the node in routes and requests."""
    x: float 
    """The x coordinate of the node in a 2D plane."""
    y: float 
    """The y coordinate of the node in a 2D plane."""
    demand: int 
    """The demand at the node. Positive for pickups, negative for deliveries."""
    time_window: tuple[int, int] 
    """A tuple (earliest, latest) representing the time window for service at the node."""
    service_time: int 
    """The time required to service the node."""
    pickup_index: int  
    """An index referencing the associated pickup node. 0 if this node is a pickup or depot."""
    delivery_index: int  
    """An index referencing the associated delivery node. 0 if this node is a delivery or depot."""
    def __str__(self) -> str:
        return (f"Node(Index: {self.index}, Demand: {self.demand}, "
                f"Time Window: {self.time_window}, Service Time: {self.service_time}, "
                f"Pickup Index: {self.pickup_index}, Delivery Index: {self.delivery_index})")
    
@dataclass
class Request:
    """A pickup and delivery request in the PDPTW problem."""
    pickup: Node 
    """The pickup node of the request."""
    delivery: Node 
    """The delivery node of the request."""
    
    @property
    def demand(self) -> int:
        """The demand of the request."""
        return self.pickup.demand  # positive demand for pickup, negative for delivery
    
    @property
    def distance(self) -> float:
        """The Euclidean distance between the pickup and delivery nodes."""
        dx = self.pickup.x - self.delivery.x
        dy = self.pickup.y - self.delivery.y
        return np.hypot(dx, dy)

    def __str__(self) -> str:
        return f"Request(Pickup: {self.pickup.index}, Delivery: {self.delivery.index}, Demand: {self.demand})"

@dataclass
class PDPTWProblem:
    """A Pickup and Delivery Problem with Time Windows (PDPTW) instance."""
    num_vehicles: int
    """The total number of vehicles available for routing."""
    vehicle_capacity: int
    """The maximum capacity of each vehicle."""
    nodes: list[Node]
    """A list of all nodes in the problem, including depot, pickups, and deliveries."""
    distance_matrix: np.ndarray = None
    """Optional pre-computed distance/time matrix. If None, will be computed from node coordinates."""
    name: str = "PDPTW Problem"
    dataset: str = "Generic"

    def __post_init__(self):
        n = len(self.nodes)

        # Only compute distance matrix if not provided
        if self.distance_matrix is None:
            self.distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = self.nodes[i].x - self.nodes[j].x
                    dy = self.nodes[i].y - self.nodes[j].y
                    self.distance_matrix[i, j] = np.hypot(dx, dy)
                
        self.nodes_dict = {node.index: node for node in self.nodes}
        self.demands = np.array([node.demand for node in self.nodes])
        self.time_windows = np.array([node.time_window for node in self.nodes])
        self.service_times = np.array([node.service_time for node in self.nodes])
        self.pickups_deliveries = self._compute_pickups_deliveries()
        self.requests = [Request(self.nodes_dict[p], self.nodes_dict[d]) for p, d in self.pickups_deliveries]
        
        self.pickup_to_delivery = {p: d for p, d in self.pickups_deliveries}
        self.delivery_to_pickup = {d: p for p, d in self.pickups_deliveries}

        # Pre-computed sets for fast lookup (O(1) instead of dict + attribute access)
        self.pickup_nodes = {p for p, d in self.pickups_deliveries}
        self.delivery_nodes = {d for p, d in self.pickups_deliveries}

        self.distance_baseline = sum(self.distance_matrix[0, i] for i in range(1, len(self.distance_matrix))) * 2
        total_demand = sum(node.demand for node in self.nodes if node.demand > 0)
        self.num_vehicles_baseline = max(1, (total_demand + self.vehicle_capacity - 1) // self.vehicle_capacity)
        
    def __str__(self) -> str:
        header = (
            f"\033[1;33mPDPTW Problem: {self.name}\033[1;30m | "
            f"\033[1;33mVehicles: {self.num_vehicles}\033[1;30m | "
            f"\033[1;33mCapacity: {self.vehicle_capacity}\033[1;30m | "
            f"\033[1;33mNodes: {len(self.nodes)}\033[1;30m | "
            f"\033[1;33mRequests: {self.num_requests}\033[0m"
        )

        # ANSI-strip for line length
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        visible_text = ansi_escape.sub('', header)
        line = "\033[1;35m" + "=" * len(visible_text) + "\033[0m"

        total_demand = sum(node.demand for node in self.nodes if node.demand > 0)
        avg_distance = self.distance_matrix[self.distance_matrix > 0].mean() if len(self.nodes) > 1 else 0

        # 2. Request distance stats
        request_distances = [r.distance for r in self.requests]
        if request_distances:
            avg_req_dist = np.mean(request_distances)
            min_req_dist = np.min(request_distances)
            max_req_dist = np.max(request_distances)
        else:
            avg_req_dist = min_req_dist = max_req_dist = 0

        # 3. Time window tightness
        tw_widths = [tw[1] - tw[0] for tw in self.time_windows]
        avg_tw_width = np.mean(tw_widths) if tw_widths else 0

        # 4. Vehicle utilization ratio
        utilization_ratio = (
            total_demand / (self.num_vehicles * self.vehicle_capacity)
            if self.num_vehicles * self.vehicle_capacity > 0 else 0
        )

        # 5. Depot spread (assuming depot is node 0)
        if len(self.nodes) > 1:
            depot_dists = self.distance_matrix[0, 1:]
            avg_depot_dist = np.mean(depot_dists)
            max_depot_dist = np.max(depot_dists)
        else:
            avg_depot_dist = max_depot_dist = 0

        stats = [
            f"\033[1;36mTotal demand:\033[0m {total_demand}",
            f"\033[1;36mBaseline vehicles:\033[0m {self.num_vehicles_baseline}",
            f"\033[1;36mVehicle utilization ratio:\033[0m {utilization_ratio:.2f}",
            f"\033[1;36mAverage pairwise distance:\033[0m {avg_distance:.2f}",
            f"\033[1;36mBaseline distance:\033[0m {self.distance_baseline:.2f}",
            "",
            f"\033[1;36mRequest distance:\033[0m avg={avg_req_dist:.2f}, min={min_req_dist:.2f}, max={max_req_dist:.2f}",
            f"\033[1;36mTime window width:\033[0m avg={avg_tw_width:.1f}",
            f"\033[1;36mDepot distance:\033[0m avg={avg_depot_dist:.2f}, max={max_depot_dist:.2f}"
        ]

        return "\n".join([line, header, line] + stats) + "\n"
    
    @property
    def num_locations(self) -> int:
        """The number of customer locations (excluding the depot). """
        return len(self.nodes) - 1  # ohne Depot
    
    @property
    def num_requests(self) -> int:
        """The number of pickup and delivery requests in the problem."""
        return len(self.pickups_deliveries)
    
    def _compute_pickups_deliveries(self) -> list[tuple[int, int]]:
        pairs = []
        seen = set()
        for node in self.nodes:
            if node.pickup_index == 0 and node.delivery_index != 0: 
                pairs.append((node.index, node.delivery_index))
        return pairs

    def is_pickup(self, node_index: int) -> bool:
        """Returns True if the node is a pickup node."""
        return node_index in self.pickup_nodes

    def is_delivery(self, node_index: int) -> bool:
        """Returns True if the node is a delivery node."""
        return node_index in self.delivery_nodes
    
    def get_pair(self, node_index: int) -> int:
        """Returns the request pair for a given pickup or delivery node."""
        node = self.nodes_dict[node_index]
        if self.is_pickup(node_index):
            return (node.index, node.delivery_index)
        elif self.is_delivery(node_index):
            return (node.pickup_index, node.index)
        else:
            raise ValueError(f"Node {node_index} is neither pickup nor delivery.")
        
    def get_other(self, node_index: int) -> int:
        """Returns the paired node index for a given pickup or delivery node."""
        if self.is_pickup(node_index):
            return self.pickup_to_delivery[node_index]
        elif self.is_delivery(node_index):
            return self.delivery_to_pickup[node_index]
        else:
            raise ValueError(f"Node {node_index} is neither pickup nor delivery.")
    
    @property
    def data(self) -> dict:
        """Returns a dictionary representation of the problem data.
            
            num_vehicles: int \n
            vehicle_capacity: int \n
            depot: int (always 0) \n
            distance_matrix: np.array \n
            demands: np.array \n
            time_windows: np.array \n
            service_times: np.array \n
            pickups_deliveries: list of (pickup_index, delivery_index) tuples
        """
        return {
            'num_vehicles': self.num_vehicles,
            'vehicle_capacity': self.vehicle_capacity,
            'depot': 0,
            'distance_matrix': self.distance_matrix,
            'demands': self.demands,
            'time_windows': self.time_windows,
            'service_times': self.service_times,
            'pickups_deliveries': self.pickups_deliveries
        }
