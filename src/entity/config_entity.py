from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_files_path: Path

@dataclass
class DataTransformConfig:
    root_dir: Path
    raw_members_data_path: Path
    raw_transactions_data_path: Path
    raw_user_logs_data_path: Path
    raw_train_data_path: Path
    final_X_train_data_path: Path
    final_X_test_data_path: Path
    y_train_data_path: Path
    y_test_data_path: Path

@dataclass
class ModelTrainConfig:
    root_dir: Path
    X_train_data_path: Path
    X_test_data_path: Path
    y_train_data_path: Path
    y_test_data_path: Path
    final_model_path: Path

@dataclass
class EvaluationConfig:
    root_dir: Path
    final_model_path: Path
    evaluation_X_path: Path
    evaluation_y_path: Path
    accuracy_txt_path: Path