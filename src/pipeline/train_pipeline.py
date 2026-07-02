from src.components.data_ingestions import DataIngestions
from src.config.manager import ConfigManager
from pathlib import Path
if __name__ == "__main__":
    
    manager = ConfigManager(Path('configs\config.yaml'), Path('configs\params.yaml'))
    ingestion_config = manager.get_ingestion_config()
    ingestion_step = DataIngestions(ingestion_config, manager.params)
    ingestion_artefact = ingestion_step.start_ingestion()
    