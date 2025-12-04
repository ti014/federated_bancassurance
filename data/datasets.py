"""Public datasets cho Lapse Prediction.

Hỗ trợ download và load các public datasets từ Kaggle và các nguồn khác.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import os
import warnings

warnings.filterwarnings('ignore')


# URLs và thông tin về public datasets
PUBLIC_DATASETS = {
    'insurance_churn': {
        'name': 'Insurance Customer Churn Dataset',
        'source': 'Kaggle',
        'url': 'https://www.kaggle.com/datasets/arashnic/churn-modeling',
        'description': 'Customer churn prediction dataset có thể adapt cho lapse prediction',
        'file_name': 'Churn_Modelling.csv',
    },
    'bank_churn': {
        'name': 'Bank Customer Churn',
        'source': 'Kaggle',
        'url': 'https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction',
        'description': 'Bank customer churn - có thể adapt cho bancassurance',
        'file_name': 'Churn_Modelling.csv',
    },
    'telco_churn': {
        'name': 'Telco Customer Churn',
        'source': 'Kaggle',
        'url': 'https://www.kaggle.com/datasets/blastchar/telco-customer-churn',
        'description': 'Telco customer churn - phù hợp để adapt',
        'file_name': 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
    },
}


def download_kaggle_dataset(
    dataset_name: str,
    output_dir: str = 'data/raw',
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
) -> str:
    """Download dataset từ Kaggle.
    
    Args:
        dataset_name: Tên dataset (format: username/dataset-name)
        output_dir: Thư mục để save dataset
        kaggle_username: Kaggle username (hoặc từ environment variable)
        kaggle_key: Kaggle API key (hoặc từ environment variable)
    
    Returns:
        Đường dẫn đến file đã download
    
    Note:
        Cần cài kaggle API: pip install kaggle
        Cần setup Kaggle credentials: https://www.kaggle.com/docs/api
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "Kaggle API not installed. Install with: pip install kaggle\n"
            "Then setup credentials: https://www.kaggle.com/docs/api"
        )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    kaggle.api.dataset_download_files(
        dataset_name,
        path=str(output_path),
        unzip=True,
    )
    
    print(f"✓ Dataset downloaded to {output_path}")
    return str(output_path)


def load_insurance_churn_dataset(
    file_path: Optional[str] = None,
    data_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load Insurance Customer Churn dataset.
    
    Args:
        file_path: Đường dẫn đến CSV file
        data_dir: Thư mục chứa data nếu file_path không được cung cấp
    
    Returns:
        Tuple của (features DataFrame, target Series)
    """
    if file_path is None:
        # Try to find file in data directory
        data_path = Path(data_dir)
        possible_files = [
            'Churn_Modelling.csv',
            'insurance_churn.csv',
            'churn_data.csv',
        ]
        
        for filename in possible_files:
            file_path = data_path / filename
            if file_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"Dataset file not found. Please download from:\n"
                f"{PUBLIC_DATASETS['insurance_churn']['url']}\n"
                f"Or provide file_path parameter."
            )
    
    df = pd.read_csv(file_path)
    
    # Common column names for churn datasets
    # Target column might be: 'Exited', 'Churn', 'churn', 'target', etc.
    target_cols = ['Exited', 'Churn', 'churn', 'Churn_Value', 'target']
    target_col = None
    
    for col in target_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Assume last column is target
        target_col = df.columns[-1]
        print(f"Warning: Using last column '{target_col}' as target")
    
    target = df[target_col]
    features = df.drop(columns=[target_col])
    
    # Drop ID columns if present
    id_cols = ['RowNumber', 'CustomerId', 'id', 'ID']
    for col in id_cols:
        if col in features.columns:
            features = features.drop(columns=[col])
    
    return features, target


def load_bank_churn_dataset(
    file_path: Optional[str] = None,
    data_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load Bank Customer Churn dataset.
    
    Args:
        file_path: Đường dẫn đến CSV file
        data_dir: Thư mục chứa data
    
    Returns:
        Tuple của (features DataFrame, target Series)
    """
    if file_path is None:
        data_path = Path(data_dir)
        possible_files = [
            'Bank Customer Churn Prediction.csv',
            'bank_churn.csv',
            'churn_data.csv',
        ]
        
        for filename in possible_files:
            file_path = data_path / filename
            if file_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"Dataset file not found. Please download from:\n"
                f"{PUBLIC_DATASETS['bank_churn']['url']}"
            )
    
    df = pd.read_csv(file_path)
    
    # Find target column
    target_cols = ['Exited', 'Churn', 'churn', 'target']
    target_col = None
    
    for col in target_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
    
    target = df[target_col]
    features = df.drop(columns=[target_col])
    
    # Drop ID columns
    id_cols = ['RowNumber', 'CustomerId', 'id', 'ID', 'customer_id']
    for col in id_cols:
        if col in features.columns:
            features = features.drop(columns=[col])
    
    return features, target


def load_telco_churn_dataset(
    file_path: Optional[str] = None,
    data_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load Telco Customer Churn dataset.
    
    Args:
        file_path: Đường dẫn đến CSV file
        data_dir: Thư mục chứa data
    
    Returns:
        Tuple của (features DataFrame, target Series)
    """
    if file_path is None:
        data_path = Path(data_dir)
        possible_files = [
            'WA_Fn-UseC_-Telco-Customer-Churn.csv',
            'telco_churn.csv',
            'churn_data.csv',
        ]
        
        for filename in possible_files:
            file_path = data_path / filename
            if file_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"Dataset file not found. Please download from:\n"
                f"{PUBLIC_DATASETS['telco_churn']['url']}"
            )
    
    df = pd.read_csv(file_path)
    
    # Handle TotalCharges: convert to numeric, handle empty strings
    if 'TotalCharges' in df.columns:
        # Replace empty strings with NaN, then convert to numeric
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill NaN with 0 (for new customers with no charges yet)
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Telco dataset usually has 'Churn' column
    if 'Churn' in df.columns:
        target = df['Churn'].map({'Yes': 1, 'No': 0})
        features = df.drop(columns=['Churn'])
    else:
        target = df.iloc[:, -1]
        features = df.iloc[:, :-1]
    
    # Drop customer ID if present
    if 'customerID' in features.columns:
        features = features.drop(columns=['customerID'])
    
    return features, target


def list_available_datasets() -> Dict:
    """List tất cả available public datasets.
    
    Returns:
        Dictionary với thông tin về các datasets
    """
    return PUBLIC_DATASETS


def load_public_dataset(
    dataset_name: str,
    file_path: Optional[str] = None,
    data_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load public dataset by name.
    
    Args:
        dataset_name: Tên dataset ('insurance_churn', 'bank_churn', 'telco_churn')
        file_path: Đường dẫn đến file (optional)
        data_dir: Thư mục chứa data
    
    Returns:
        Tuple của (features DataFrame, target Series)
    """
    loaders = {
        'insurance_churn': load_insurance_churn_dataset,
        'bank_churn': load_bank_churn_dataset,
        'telco_churn': load_telco_churn_dataset,
    }
    
    if dataset_name not in loaders:
        available = ', '.join(loaders.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    
    return loaders[dataset_name](file_path=file_path, data_dir=data_dir)

