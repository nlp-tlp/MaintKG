"""Utilities for path handling in the project."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory (the directory containing src/).

    Returns:
        Path: The absolute path to the project root directory.
    """
    current_path = Path(__file__).resolve()
    # Navigate up from utils/path_utils.py to project root
    # utils/path_utils.py -> utils -> maintkg -> src -> project_root
    return current_path.parent.parent.parent.parent


def get_cache_dir() -> Path:
    """Get the cache directory path.

    Returns:
        Path: The absolute path to the cache directory.
    """
    return get_project_root() / "cache"


def get_cache_file(filename: str) -> str:
    """Get a cache file path as string.

    Args:
        filename: Name of the cache file without extension

    Returns:
        str: The absolute path to the cache file as a string
    """
    cache_path = get_cache_dir() / filename
    return str(cache_path)


def get_output_dir() -> Path:
    """Get the output directory path.

    Returns:
        Path: The absolute path to the output directory.
    """
    return get_project_root() / "output"


def get_input_dir() -> Path:
    """Get the input directory path.

    Returns:
        Path: The absolute path to the input directory.
    """
    return get_project_root() / "input"


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: The directory path to ensure exists.
    """
    path.mkdir(parents=True, exist_ok=True)


# Initialise directories on import
ensure_dir(get_cache_dir())
ensure_dir(get_output_dir())
ensure_dir(get_input_dir())
