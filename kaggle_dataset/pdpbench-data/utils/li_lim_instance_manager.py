from pathlib import Path
from typing import Optional, Iterator
from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_reader import pdptw_reader


class LiLimInstanceManager:
    """
    Manager for Li & Lim PDPTW Benchmark instances.
    Enables navigation and iteration through instances of different sizes and categories.
    """

    # Available sizes
    SIZES = [100, 200, 400, 600, 1000]

    # Instance names per category (for size 100)
    CATEGORIES = {
        'lc1': ['lc101', 'lc102', 'lc103', 'lc104', 'lc105', 'lc106', 'lc107', 'lc108', 'lc109'],
        'lc2': ['lc201', 'lc202', 'lc203', 'lc204', 'lc205', 'lc206', 'lc207', 'lc208'],
        'lr1': ['lr101', 'lr102', 'lr103', 'lr104', 'lr105', 'lr106', 'lr107', 'lr108', 'lr109', 'lr110', 'lr111', 'lr112'],
        'lr2': ['lr201', 'lr202', 'lr203', 'lr204', 'lr205', 'lr206', 'lr207', 'lr208', 'lr209', 'lr210', 'lr211'],
        'lrc1': ['lrc101', 'lrc102', 'lrc103', 'lrc104', 'lrc105', 'lrc106', 'lrc107', 'lrc108'],
        'lrc2': ['lrc201', 'lrc202', 'lrc203', 'lrc204', 'lrc205', 'lrc206', 'lrc207', 'lrc208']
    }

    def __init__(self, base_dir: str = 'data'):
        """
        Args:
            base_dir: Base directory with pdp_100, pdp_200, etc. subdirectories
        """
        self.base_dir = Path(base_dir)
        self.current_size = 100
        self.current_category = 'lc1'
        self.current_index = 0
        self._problem_cache = {}

    def _get_categories(self, size: int) -> dict:
        """
        Returns the category structure for a given size.

        For size 100: uses lowercase names (lc101, lr101, etc.)
        For other sizes: uses uppercase names (LC1_2_1, LR1_2_1, etc.)
        """
        if size == 100:
            return {
                'lc1': ['lc101', 'lc102', 'lc103', 'lc104', 'lc105', 'lc106', 'lc107', 'lc108', 'lc109'],
                'lc2': ['lc201', 'lc202', 'lc203', 'lc204', 'lc205', 'lc206', 'lc207', 'lc208'],
                'lr1': ['lr101', 'lr102', 'lr103', 'lr104', 'lr105', 'lr106', 'lr107', 'lr108', 'lr109', 'lr110', 'lr111', 'lr112'],
                'lr2': ['lr201', 'lr202', 'lr203', 'lr204', 'lr205', 'lr206', 'lr207', 'lr208', 'lr209', 'lr210', 'lr211'],
                'lrc1': ['lrc101', 'lrc102', 'lrc103', 'lrc104', 'lrc105', 'lrc106', 'lrc107', 'lrc108'],
                'lrc2': ['lrc201', 'lrc202', 'lrc203', 'lrc204', 'lrc205', 'lrc206', 'lrc207', 'lrc208']
            }
        else:
            size_code = size // 100
            return {
                'lc1': [f'LC1_{size_code}_{i}' for i in range(1, 11)],
                'lc2': [f'LC2_{size_code}_{i}' for i in range(1, 11)],
                'lr1': [f'LR1_{size_code}_{i}' for i in range(1, 11)],
                'lr2': [f'LR2_{size_code}_{i}' for i in range(1, 11)],
                'lrc1': [f'LRC1_{size_code}_{i}' for i in range(1, 11)],
                'lrc2': [f'LRC2_{size_code}_{i}' for i in range(1, 11)]
            }

    def _get_path(self, instance_name: str, size: int) -> Path:
        """Returns path to instance file."""
        return self.base_dir / f'pdp_{size}' / f'{instance_name}.txt'

    def load(self, instance_name: str, size: Optional[int] = None) -> PDPTWProblem:
        """
        Loads a specific instance.

        Args:
            instance_name: Instance name (e.g., 'lc101' or 'LC1_2_1')
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
        categories = self._get_categories(self.current_size)
        instance_name = categories[self.current_category][self.current_index]
        return self.load(instance_name, self.current_size)

    def next(self) -> PDPTWProblem:
        """
        Jumps to the next instance in the current category and size.
        Automatically switches to the next category when at the end.
        """
        categories_dict = self._get_categories(self.current_size)
        category_instances = categories_dict[self.current_category]
        self.current_index += 1

        if self.current_index >= len(category_instances):
            # Move to next category
            categories = list(categories_dict.keys())
            current_cat_idx = categories.index(self.current_category)

            if current_cat_idx < len(categories) - 1:
                self.current_category = categories[current_cat_idx + 1]
                self.current_index = 0
            else:
                # Wrap back to start
                self.current_category = categories[0]
                self.current_index = 0

        return self.current()

    def prev(self) -> PDPTWProblem:
        """Jumps to the previous instance."""
        self.current_index -= 1

        if self.current_index < 0:
            # Move to previous category
            categories_dict = self._get_categories(self.current_size)
            categories = list(categories_dict.keys())
            current_cat_idx = categories.index(self.current_category)

            if current_cat_idx > 0:
                self.current_category = categories[current_cat_idx - 1]
                self.current_index = len(categories_dict[self.current_category]) - 1
            else:
                # Wrap to end
                self.current_category = categories[-1]
                self.current_index = len(categories_dict[self.current_category]) - 1

        return self.current()

    def jump_to_size(self, size: int) -> 'LiLimInstanceManager':
        """
        Switches to the specified problem size.

        Args:
            size: One of the available sizes (100, 200, 400, 600, 1000)
        """
        if size not in self.SIZES:
            raise ValueError(f"Size {size} not available. Choose from {self.SIZES}")
        self.current_size = size
        return self

    def jump_to_category(self, category: str) -> 'LiLimInstanceManager':
        """
        Switches to the specified category.

        Args:
            category: One of the categories ('lc1', 'lc2', 'lr1', 'lr2', 'lrc1', 'lrc2')
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Category {category} not available. Choose from {list(self.CATEGORIES.keys())}")
        self.current_category = category
        self.current_index = 0
        return self

    def jump_to(self, instance_name: str) -> 'LiLimInstanceManager':
        """
        Jumps directly to a named instance.

        Args:
            instance_name: Instance name (e.g., 'lc201' or 'LC1_2_1')
        """
        categories_dict = self._get_categories(self.current_size)
        for category, instances in categories_dict.items():
            if instance_name in instances:
                self.current_category = category
                self.current_index = instances.index(instance_name)
                return self

        raise ValueError(f"Instance {instance_name} not found for size {self.current_size}")

    def get_all_in_category(self, category: str, size: Optional[int] = None) -> list[PDPTWProblem]:
        """
        Loads all instances of a category.

        Args:
            category: Category name
            size: Problem size (default: current_size)
        """
        if size is None:
            size = self.current_size

        categories_dict = self._get_categories(size)
        if category not in categories_dict:
            raise ValueError(f"Unknown category: {category}")

        problems = []
        for instance_name in categories_dict[category]:
            problems.append(self.load(instance_name, size))

        return problems

    def get_all(self, size: Optional[int] = None) -> list[PDPTWProblem]:
        """
        Loads all available instances of a size.

        Args:
            size: Problem size (default: current_size)
        """
        if size is None:
            size = self.current_size

        categories_dict = self._get_categories(size)
        problems = []
        for category in categories_dict.keys():
            problems.extend(self.get_all_in_category(category, size))

        return problems

    def iterate_current(self) -> Iterator[tuple[str, int, PDPTWProblem]]:
        """
        Iterates only over instances of the current size and category.

        Yields:
            (instance_name, size, problem) tuples
        """
        categories_dict = self._get_categories(self.current_size)
        for instance_name in categories_dict[self.current_category]:
            problem = self.load(instance_name, self.current_size)
            yield (instance_name, self.current_size, problem)

    def iterate_category(self, category: str, size: Optional[int] = None) -> Iterator[tuple[str, int, PDPTWProblem]]:
        """
        Iterates over all instances of a category.

        Args:
            category: Category name
            size: Problem size (default: current_size)

        Yields:
            (instance_name, size, problem) tuples
        """
        if size is None:
            size = self.current_size

        categories_dict = self._get_categories(size)
        for instance_name in categories_dict[category]:
            problem = self.load(instance_name, size)
            yield (instance_name, size, problem)

    def iterate_size(self, size: int) -> Iterator[tuple[str, str, int, PDPTWProblem]]:
        """
        Iterates over all instances of ALL categories for a specific size.

        Args:
            size: Problem size

        Yields:
            (instance_name, category, size, problem) tuples
        """
        categories_dict = self._get_categories(size)
        for category, instances in categories_dict.items():
            for instance_name in instances:
                problem = self.load(instance_name, size)
                yield (instance_name, category, size, problem)

    def iterate_all(self,
                   sizes: Optional[list[int]] = None,
                   categories: Optional[list[str]] = None) -> Iterator[tuple[str, str, int, PDPTWProblem]]:
        """
        Iterates over ALL instances across all sizes and categories.

        Args:
            sizes: List of sizes (default: all available)
            categories: List of categories (default: all available)

        Yields:
            (instance_name, category, size, problem) tuples
        """
        if sizes is None:
            sizes = self.SIZES
        if categories is None:
            categories = list(self.CATEGORIES.keys())

        for size in sizes:
            categories_dict = self._get_categories(size)
            for category in categories:
                if category not in categories_dict:
                    continue
                for instance_name in categories_dict[category]:
                    try:
                        problem = self.load(instance_name, size)
                        yield (instance_name, category, size, problem)
                    except FileNotFoundError:
                        continue  # Skip missing files

    def __repr__(self) -> str:
        categories = self._get_categories(self.current_size)
        instance_name = categories[self.current_category][self.current_index]
        return f"LiLimInstanceManager(current={instance_name}, size={self.current_size}, category={self.current_category})"
