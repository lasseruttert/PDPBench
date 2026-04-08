from .li_lim_reader import li_lim_reader
from .mendeley_reader import mendeley_reader

def pdptw_reader(file_path):
    """Smart reader that automatically detects the PDPTW file format and uses the appropriate parser.

    Supports:
    - Li & Lim format (simple numeric data, no metadata headers)
    - Mendeley/Sartori & Buriol format (metadata headers with NAME, LOCATION, etc.)

    Args:
        file_path: Path to the PDPTW instance file

    Returns:
        PDPTWProblem: The parsed PDPTW problem instance
    """
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()

    # Mendeley format has metadata headers like "NAME:", "LOCATION:", etc.
    # Li & Lim format starts with numeric data: "num_vehicles capacity speed"
    if ':' in first_line and any(keyword in first_line for keyword in ['NAME', 'LOCATION', 'TYPE', 'COMMENT']):
        # Mendeley format detected
        return mendeley_reader(file_path)
    else:
        # Li & Lim format detected
        return li_lim_reader(file_path)

if __name__ == "__main__":
    # Test with Li & Lim format
    print("Testing Li & Lim format:")
    problem1 = pdptw_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc101.txt')
    print(f"Loaded {problem1.num_requests} requests, {problem1.num_vehicles} vehicles\n")

    # Test with Mendeley format
    print("Testing Mendeley format:")
    problem2 = pdptw_reader('G:/Meine Ablage/rl-memetic-pdptw/data/n100/bar-n100-1.txt')
    print(f"Loaded {problem2.num_requests} requests, {problem2.num_vehicles} vehicles")
