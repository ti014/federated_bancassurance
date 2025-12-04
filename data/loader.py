"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def load_dataset(
    file_path: Optional[str] = None,
    use_synthetic: bool = False,
    dataset_name: Optional[str] = None,
    n_samples: int = 10000,
    random_state: int = 42,
    data_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load insurance dataset từ file, public dataset, hoặc generate synthetic data.
    
    Args:
        file_path: Đường dẫn đến CSV file (nếu None thì tìm dataset hoặc synthetic)
        use_synthetic: Nếu True, generate synthetic data (chỉ dùng khi không có dataset)
        dataset_name: Tên public dataset ('insurance_churn', 'bank_churn', 'telco_churn')
        n_samples: Số samples nếu dùng synthetic
        random_state: Random seed
        data_dir: Thư mục chứa data
    
    Returns:
        Tuple của (features DataFrame, target Series)
    
    Examples:
        # Load từ file
        features, target = load_dataset(file_path='data/raw/insurance.csv')
        
        # Load public dataset
        features, target = load_dataset(dataset_name='insurance_churn')
        
        # Load synthetic (fallback)
        features, target = load_dataset(use_synthetic=True, n_samples=10000)
    """
    # Priority 1: Load từ file_path nếu được cung cấp
    if file_path is not None:
        try:
            df = pd.read_csv(file_path)
            # Assume last column is target
            target = df.iloc[:, -1]
            features = df.iloc[:, :-1]
            print(f"[OK] Loaded dataset from {file_path}")
            return features, target
        except FileNotFoundError:
            print(f"[WARNING] File not found: {file_path}")
            if not use_synthetic:
                print("  Trying to load public dataset...")
    
    # Priority 2: Load public dataset nếu được chỉ định
    if dataset_name is not None:
        try:
            from .datasets import load_public_dataset
            features, target = load_public_dataset(
                dataset_name=dataset_name,
                data_dir=data_dir,
            )
            print(f"[OK] Loaded public dataset: {dataset_name}")
            return features, target
        except FileNotFoundError as e:
            print(f"[WARNING] Public dataset not found: {e}")
            if not use_synthetic:
                print("  Falling back to synthetic data...")
    
    # Priority 3: Try to find any dataset file in data/raw
    # Ưu tiên telco_churn.csv > bank_churn.csv > các file khác
    if not use_synthetic:
        data_path = Path(data_dir)
        if data_path.exists():
            # Priority order: telco_churn > bank_churn > others
            priority_files = ['telco_churn.csv', 'bank_churn.csv']
            csv_files = list(data_path.glob('*.csv'))
            
            # Try priority files first
            file_path = None
            for priority_file in priority_files:
                priority_path = data_path / priority_file
                if priority_path.exists():
                    file_path = priority_path
                    break
            
            # If no priority file found, use first CSV file
            if file_path is None and csv_files:
                file_path = csv_files[0]
            
            if file_path:
                print(f"[OK] Found dataset file: {file_path}")
                try:
                    # Try to use dataset loader if it's a known dataset
                    if 'telco_churn' in str(file_path):
                        from .datasets import load_telco_churn_dataset
                        return load_telco_churn_dataset(file_path=str(file_path))
                    elif 'bank_churn' in str(file_path):
                        from .datasets import load_bank_churn_dataset
                        return load_bank_churn_dataset(file_path=str(file_path))
                    else:
                        # Generic CSV loader
                        df = pd.read_csv(file_path)
                        target = df.iloc[:, -1]
                        features = df.iloc[:, :-1]
                        return features, target
                except Exception as e:
                    print(f"[WARNING] Error loading {file_path}: {e}")
    
    # Priority 4: Generate synthetic data (fallback)
    if use_synthetic:
        print("[WARNING] Using synthetic data (not recommended for final report)")
        print("  Consider downloading a public dataset:")
        print("  - python scripts/download_datasets.py --list")
        from .synthetic import generate_synthetic_insurance_data
        return generate_synthetic_insurance_data(
            n_samples=n_samples,
            random_state=random_state,
        )
    
    # If we get here, no data source was found
    raise ValueError(
        "No dataset found. Please:\n"
        "1. Provide file_path parameter, or\n"
        "2. Set dataset_name to load public dataset, or\n"
        "3. Place a CSV file in data/raw/, or\n"
        "4. Set use_synthetic=True (not recommended)\n"
        "\nTo see available public datasets:\n"
        "  python scripts/download_datasets.py --list"
    )


def preprocess_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numerical: bool = True,
    encode_categorical: bool = True,
) -> Dict[str, Any]:
    """Preprocess data: encoding, scaling, train/test split.
    
    Args:
        features: Features DataFrame
        target: Target Series
        test_size: Tỷ lệ test set
        random_state: Random seed
        scale_numerical: Có scale numerical features không
        encode_categorical: Có encode categorical features không
    
    Returns:
        Dictionary chứa:
        - X_train, X_test: Processed features
        - y_train, y_test: Targets
        - scaler: Fitted scaler (nếu dùng)
        - encoders: Dictionary của encoders (nếu dùng)
        - feature_names: Tên các features sau khi process
    """
    X = features.copy()
    y = target.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    encoders = {}
    
    # Encode categorical variables
    if encode_categorical and len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # Scale numerical features
    scaler = None
    if scale_numerical and len(numerical_cols) > 0:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': X.columns.tolist(),
        'n_features': X.shape[1],
    }

