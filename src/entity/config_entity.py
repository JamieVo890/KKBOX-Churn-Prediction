from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_members_data_path: Path
    raw_transactions_data_path: Path
    raw_user_logs_data_path: Path