import yaml
from pathlib import Path
import numpy as np
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads yaml file and returns a ConfigBox.

    Args:
        path_to_yaml (Path): path to the yaml file

    Raises:
        ValueError: if yaml file is empty
        e: any other exception during file read

    Returns:
        ConfigBox: config as a ConfigBox object
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as file:
            content = yaml.safe_load(file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
def write_npy(arr: np.ndarray, path_to_file: Path) -> None:
    """Save a numpy array to a .npy file.

    Args:
        arr (np.ndarray): array to save.
        path_to_file (Path): path where the .npy file will be written.

    Raises:
        e: any exception raised during file writing.

    Returns:
        None
    """
    try:
        with open(path_to_file, "wb") as file:
            np.save(file, arr)
    except Exception as e:
        raise e
    
    
       