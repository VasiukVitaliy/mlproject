from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    data_path: Path
    train_path: Path
    test_path: Path
