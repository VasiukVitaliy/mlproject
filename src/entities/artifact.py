from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    train_path: Path
    test_path: Path