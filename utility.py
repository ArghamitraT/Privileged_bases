import os
import time


def create_timestamped_filename(kind: str) -> str:
    """
    Create a timestamped filename with the given kind prefix.

    Args:
        kind (str): The prefix for the filename (e.g., 'log', 'data')

    Returns:
        str: A filename in the format '{kind}_YYYY_MM_DD__HH_MM_SS'
    """
    trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")
    return f"{kind}_{trimester}"



def get_path(target_suffix: str, root_name: str = "Mat_embedding_hyperbole") -> str:
    """
    Get the absolute path inside the 'Mat_embedding_hyperbole' project,
    given a relative suffix (e.g., 'code/figure').
    - Finds the project root by walking upward from the current directory.
    - Creates the target folder if it doesn't exist.
    
    Args:
        target_suffix (str): Relative path inside the project (e.g. 'code/figure')
        root_name (str): Name of the project root folder (default = 'Mat_embedding_hyperbole')
    
    Returns:
        str: Absolute path to the target directory (created if not existing)
    
    Raises:
        FileNotFoundError: If the project root cannot be found upward from cwd.
    """
    cwd = os.path.abspath(os.getcwd())
    parts = cwd.split(os.sep)

    # Step 1: Find the project root folder
    root_path = None
    for i in range(len(parts), 0, -1):
        candidate = os.sep.join(parts[:i])
        if os.path.basename(candidate) == root_name:
            root_path = candidate
            break

    if root_path is None:
        raise FileNotFoundError(f"Project root folder '{root_name}' not found upward from {cwd}")

    # Step 2: Construct the full target path
    target_path = os.path.join(root_path, os.path.normpath(target_suffix))

    # Step 3: Create folder if missing
    os.makedirs(target_path, exist_ok=True)

    return target_path



# Example usage:
if __name__ == "__main__":
    # print(find_project_path("Mat_embedding_hyperbole/code/figure"))
    target_path = get_project_path("code/figure")
    print(target_path)