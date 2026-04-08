from pathlib import Path
from typing import Optional, Iterator
from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_reader import pdptw_reader
import re


class MendeleyInstanceManager:
    """
    Manager for Mendeley/Sartori & Buriol real-world PDPTW instances.
    Enables navigation and iteration through instances by city and size.
    """

    def __init__(self, base_dir: str = 'data'):
        """
        Args:
            base_dir: Base directory containing n* subdirectories (e.g., n100/)
        """
        self.base_dir = Path(base_dir)

        # Discover available instances
        self.sizes, self.cities = self._discover_instances()

        # Current navigation state
        self.current_size = self.sizes[0] if self.sizes else 100
        self.current_city = list(self.cities.keys())[0] if self.cities else None
        self.current_index = 0

        # Cache
        self._problem_cache = {}

    def _discover_instances(self) -> tuple[list[int], dict[str, dict[int, list[str]]]]:
        """
        Discovers available Mendeley instances from the filesystem.

        Returns:
            (sizes, cities) where:
            - sizes: List of available sizes (e.g., [100])
            - cities: Dict mapping city -> {size -> [instance_names]}
        """
        sizes = []
        cities = {}

        # Find all n* directories
        for dir_path in self.base_dir.glob('n*'):
            if not dir_path.is_dir():
                continue

            # Extract size from directory name (e.g., n100 -> 100)
            match = re.match(r'n(\d+)', dir_path.name)
            if not match:
                continue

            size = int(match.group(1))
            if size not in sizes:
                sizes.append(size)

            # Find all instance files in this directory
            for file_path in dir_path.glob('*.txt'):
                # Parse filename: {city}-n{size}-{num}.txt
                match = re.match(r'([a-z]+)-n\d+-(\d+)\.txt', file_path.name)
                if not match:
                    continue

                city = match.group(1)
                instance_name = file_path.stem  # filename without .txt

                # Initialize city structure
                if city not in cities:
                    cities[city] = {}
                if size not in cities[city]:
                    cities[city][size] = []

                # Add instance
                if instance_name not in cities[city][size]:
                    cities[city][size].append(instance_name)

        # Sort for consistency
        sizes.sort()
        for city in cities:
            for size in cities[city]:
                cities[city][size].sort()

        return sizes, cities

    def _get_path(self, instance_name: str, size: int) -> Path:
        """Returns path to instance file."""
        return self.base_dir / f'n{size}' / f'{instance_name}.txt'

    def load(self, instance_name: str, size: Optional[int] = None) -> PDPTWProblem:
        """
        Loads a specific instance.

        Args:
            instance_name: Instance name (e.g., 'bar-n100-1')
            size: Problem size (default: current_size)

        Returns:
            Loaded PDPTW problem instance
        """
        if size is None:
            size = self.current_size

        cache_key = (instance_name, size)
        if cache_key not in self._problem_cache:
            path = self._get_path(instance_name, size)
            if not path.exists():
                raise FileNotFoundError(f"Instance file not found: {path}")
            self._problem_cache[cache_key] = pdptw_reader(str(path))

        return self._problem_cache[cache_key]

    def current(self) -> PDPTWProblem:
        """Returns the current problem."""
        if not self.current_city or self.current_city not in self.cities:
            raise ValueError("No valid city selected")

        if self.current_size not in self.cities[self.current_city]:
            raise ValueError(f"Size {self.current_size} not available for city {self.current_city}")

        instances = self.cities[self.current_city][self.current_size]
        if self.current_index >= len(instances):
            self.current_index = 0

        instance_name = instances[self.current_index]
        return self.load(instance_name, self.current_size)

    def next(self) -> PDPTWProblem:
        """
        Jumps to the next instance in the current city and size.
        Automatically switches to the next city when at the end.
        """
        if not self.current_city:
            raise ValueError("No city selected")

        instances = self.cities[self.current_city][self.current_size]
        self.current_index += 1

        if self.current_index >= len(instances):
            # Move to next city
            city_list = sorted(self.cities.keys())
            current_city_idx = city_list.index(self.current_city)

            if current_city_idx < len(city_list) - 1:
                self.current_city = city_list[current_city_idx + 1]
                self.current_index = 0
            else:
                # Wrap back to first city
                self.current_city = city_list[0]
                self.current_index = 0

        return self.current()

    def prev(self) -> PDPTWProblem:
        """Jumps to the previous instance."""
        if not self.current_city:
            raise ValueError("No city selected")

        self.current_index -= 1

        if self.current_index < 0:
            # Move to previous city
            city_list = sorted(self.cities.keys())
            current_city_idx = city_list.index(self.current_city)

            if current_city_idx > 0:
                self.current_city = city_list[current_city_idx - 1]
                self.current_index = len(self.cities[self.current_city][self.current_size]) - 1
            else:
                # Wrap to last city
                self.current_city = city_list[-1]
                self.current_index = len(self.cities[self.current_city][self.current_size]) - 1

        return self.current()

    def jump_to_size(self, size: int) -> 'MendeleyInstanceManager':
        """
        Switches to the specified problem size.

        Args:
            size: One of the available sizes
        """
        if size not in self.sizes:
            raise ValueError(f"Size {size} not available. Choose from {self.sizes}")
        self.current_size = size
        self.current_index = 0
        return self

    def jump_to_city(self, city: str) -> 'MendeleyInstanceManager':
        """
        Switches to the specified city.

        Args:
            city: City name (e.g., 'bar', 'ber', 'nyc', 'poa')
        """
        if city not in self.cities:
            raise ValueError(f"City '{city}' not available. Choose from {list(self.cities.keys())}")
        self.current_city = city
        self.current_index = 0
        return self

    def jump_to(self, instance_name: str) -> 'MendeleyInstanceManager':
        """
        Jumps directly to a named instance.

        Args:
            instance_name: Instance name (e.g., 'bar-n100-1')
        """
        for city, sizes_dict in self.cities.items():
            for size, instances in sizes_dict.items():
                if instance_name in instances:
                    self.current_city = city
                    self.current_size = size
                    self.current_index = instances.index(instance_name)
                    return self

        raise ValueError(f"Instance {instance_name} not found")

    def get_all_in_city(self, city: str, size: Optional[int] = None) -> list[PDPTWProblem]:
        """
        Loads all instances of a city.

        Args:
            city: City name
            size: Problem size (default: current_size)
        """
        if size is None:
            size = self.current_size

        if city not in self.cities:
            raise ValueError(f"Unknown city: {city}")

        if size not in self.cities[city]:
            raise ValueError(f"Size {size} not available for city {city}")

        problems = []
        for instance_name in self.cities[city][size]:
            problems.append(self.load(instance_name, size))

        return problems

    def get_all(self, size: Optional[int] = None) -> list[PDPTWProblem]:
        """
        Loads all available instances of a size across all cities.

        Args:
            size: Problem size (default: current_size)
        """
        if size is None:
            size = self.current_size

        problems = []
        for city in sorted(self.cities.keys()):
            if size in self.cities[city]:
                problems.extend(self.get_all_in_city(city, size))

        return problems

    def iterate_current(self) -> Iterator[tuple[str, int, PDPTWProblem]]:
        """
        Iterates only over instances of the current size and city.

        Yields:
            (instance_name, size, problem) tuples
        """
        if not self.current_city or self.current_size not in self.cities[self.current_city]:
            return

        for instance_name in self.cities[self.current_city][self.current_size]:
            problem = self.load(instance_name, self.current_size)
            yield (instance_name, self.current_size, problem)

    def iterate_city(self, city: str, size: Optional[int] = None) -> Iterator[tuple[str, str, int, PDPTWProblem]]:
        """
        Iterates over all instances in a specific city.

        Args:
            city: City name (e.g., 'bar', 'ber', 'nyc', 'poa')
            size: Problem size (default: current_size)

        Yields:
            (instance_name, city, size, problem) tuples
        """
        if city not in self.cities:
            raise ValueError(f"City '{city}' not found. Available cities: {list(self.cities.keys())}")

        if size is None:
            size = self.current_size

        if size not in self.cities[city]:
            raise ValueError(f"Size {size} not available for city '{city}'")

        for instance_name in self.cities[city][size]:
            problem = self.load(instance_name, size)
            yield (instance_name, city, size, problem)

    def iterate_all(self, sizes: Optional[list[int]] = None) -> Iterator[tuple[str, str, int, PDPTWProblem]]:
        """
        Iterates over ALL instances across all cities and sizes.

        Args:
            sizes: List of sizes to iterate (default: all available sizes)

        Yields:
            (instance_name, city, size, problem) tuples
        """
        if sizes is None:
            sizes = self.sizes

        for size in sizes:
            for city in sorted(self.cities.keys()):
                if size not in self.cities[city]:
                    continue
                for instance_name in self.cities[city][size]:
                    problem = self.load(instance_name, size)
                    yield (instance_name, city, size, problem)

    def __repr__(self) -> str:
        if not self.current_city:
            return "MendeleyInstanceManager(no instances found)"

        instances = self.cities.get(self.current_city, {}).get(self.current_size, [])
        instance_name = instances[self.current_index] if self.current_index < len(instances) else 'None'
        return f"MendeleyInstanceManager(current={instance_name}, size={self.current_size}, city={self.current_city})"
