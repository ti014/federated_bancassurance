"""Data generation và splitting cho Vertical Federated Learning.

Module này xử lý:
- Load và convert datasets thành Vertical FL format
- Split data theo features (Bank vs Insurance)
- Preprocess data cho training

Vertical FL: Chia data theo features (columns), không phải samples (rows).
- Bank data: Financial features (balance, credit_score, debt_ratio, etc.)
- Insurance data: Policy features (premium, term, coverage, etc.)
- Cùng customer IDs để match
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Package imports - rõ ràng và nhất quán
from data.datasets import load_bank_churn_dataset, load_telco_churn_dataset


def create_vertical_fl_from_bank_churn(
    bank_file: Optional[str] = None,
    data_dir: str = 'data/raw',
    save_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert Bank Customer Churn dataset thành Vertical FL format cho Bancassurance.
    
    Chia features thành:
    - Bank features: Financial và banking features (Balance, CreditScore, Age, Geography, 
                     NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Tenure)
    - Insurance features: Policy features được tạo từ banking context
                          (Premium amount dựa trên Balance, Policy term, Coverage amount, etc.)
    
    Args:
        bank_file: Path to Bank Churn CSV file
        data_dir: Directory to search for Bank file
        save_dir: Directory to save split files
    
    Returns:
        Tuple (bank_df, insurance_df)
    """
    from data.datasets import load_bank_churn_dataset
    
    # Load Bank Churn dataset
    features, target = load_bank_churn_dataset(file_path=bank_file, data_dir=data_dir)
    
    # Add customer_id và target back
    df = features.copy()
    df['customer_id'] = [f'C{i:06d}' for i in range(1, len(df) + 1)]
    
    # Map target: Exited/Churn → Lapse (1 = Lapse, 0 = No Lapse)
    if target.dtype == 'object':
        df['lapse'] = target.map({'Yes': 1, 'No': 0, 'Churn': 1, 'No Churn': 0}).fillna(target.astype(int))
    else:
        df['lapse'] = target.values
    
    # Define Bank features (Financial và Banking features)
    # Ưu tiên các features thực sự thuộc về Bank
    bank_features_candidates = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    
    # Filter chỉ lấy features có trong dataset
    bank_features = [f for f in bank_features_candidates if f in df.columns]
    
    # Tạo Insurance features từ Bank context (simulate policy features)
    # Trong thực tế, Insurance sẽ có data riêng, nhưng để simulation:
    # - Premium: Dựa trên Balance và EstimatedSalary
    # - Policy Term: Dựa trên Tenure
    # - Coverage: Dựa trên EstimatedSalary
    # - Payment Frequency: Dựa trên IsActiveMember
    # - Policy Type: Dựa trên NumOfProducts
    
    insurance_features = []
    
    # Premium amount (monthly) - dựa trên Balance và Salary
    if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
        # Premium = 0.01 * Balance + 0.0001 * Salary (normalized)
        df['premium'] = (df['Balance'] * 0.01 + df['EstimatedSalary'] * 0.0001).clip(lower=100, upper=5000)
        insurance_features.append('premium')
    
    # Policy term (years) - dựa trên Tenure
    if 'Tenure' in df.columns:
        # Policy term = tenure / 2 (years), min 1, max 20
        df['policy_term'] = (df['Tenure'] / 2).clip(lower=1, upper=20).astype(int)
        insurance_features.append('policy_term')
    
    # Coverage amount - dựa trên EstimatedSalary
    if 'EstimatedSalary' in df.columns:
        # Coverage = 10x annual salary, rounded to nearest 100k
        df['coverage'] = (df['EstimatedSalary'] * 10 / 100000).round() * 100000
        df['coverage'] = df['coverage'].clip(lower=100000, upper=5000000)
        insurance_features.append('coverage')
    
    # Payment frequency - dựa trên IsActiveMember
    if 'IsActiveMember' in df.columns:
        # 1 = Monthly, 0 = Quarterly
        df['payment_frequency'] = df['IsActiveMember']
        insurance_features.append('payment_frequency')
    
    # Policy type - dựa trên NumOfProducts
    if 'NumOfProducts' in df.columns:
        # Map: 1-2 products = Basic, 3+ = Premium
        df['policy_type'] = (df['NumOfProducts'] >= 3).astype(int)
        insurance_features.append('policy_type')
    
    # Age group (có thể Insurance cũng có)
    if 'Age' in df.columns:
        df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3]).astype(int)
        insurance_features.append('age_group')
    
    # Create Bank DataFrame
    bank_df = df[['customer_id'] + bank_features].copy()
    
    # Create Insurance DataFrame (with target)
    insurance_df = df[['customer_id'] + insurance_features + ['lapse']].copy()
    
    # Rename target column
    insurance_df = insurance_df.rename(columns={'lapse': 'churn'})
    
    # Optional: Save to CSV (chỉ save nếu save_dir được chỉ định)
    if save_dir is not None:
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        bank_file_path = output_path / 'bank_data.csv'
        insurance_file_path = output_path / 'insurance_data.csv'
        
        bank_df.to_csv(bank_file_path, index=False)
        insurance_df.to_csv(insurance_file_path, index=False)
        
        print(f"[OK] Converted Bank Churn to Vertical FL format (Bancassurance):")
        print(f"  - Bank data: {len(bank_df)} samples, {len(bank_features)} features")
        print(f"    Features: {bank_features}")
        print(f"  - Insurance data: {len(insurance_df)} samples, {len(insurance_features)} features")
        print(f"    Features: {insurance_features}")
        print(f"  - Saved to: {bank_file_path} and {insurance_file_path}")
    else:
        print(f"[OK] Converted Bank Churn to Vertical FL format (Bancassurance):")
        print(f"  - Bank data: {len(bank_df)} samples, {len(bank_features)} features")
        print(f"    Features: {bank_features}")
        print(f"  - Insurance data: {len(insurance_df)} samples, {len(insurance_features)} features")
        print(f"    Features: {insurance_features}")
        print(f"  - Generated in memory (not saved)")
    
    return bank_df, insurance_df


def create_vertical_fl_from_telco(
    telco_file: Optional[str] = None,
    data_dir: str = 'data/raw',
    save_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert Telco Churn dataset thành Vertical FL format (fallback option).
    
    Chia features thành:
    - Bank features: Financial-like features (tenure, MonthlyCharges, TotalCharges, demographics)
    - Insurance features: Service features (PhoneService, InternetService, Contract, etc.)
    
    Args:
        telco_file: Path to Telco CSV file
        data_dir: Directory to search for Telco file
        save_dir: Directory to save split files
    
    Returns:
        Tuple (bank_df, insurance_df)
    """
    from data.datasets import load_telco_churn_dataset
    
    # Load Telco dataset
    features, target = load_telco_churn_dataset(file_path=telco_file, data_dir=data_dir)
    
    # Add customer_id và target back
    df = features.copy()
    df['customer_id'] = [f'C{i:06d}' for i in range(1, len(df) + 1)]
    df['churn'] = target.values
    
    # Define Bank features (financial-like trong Telco context)
    bank_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ]
    
    # Define Insurance features (services + policy info)
    insurance_features = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Filter chỉ lấy features có trong dataset
    bank_features = [f for f in bank_features if f in df.columns]
    insurance_features = [f for f in insurance_features if f in df.columns]
    
    # Create Bank DataFrame
    bank_df = df[['customer_id'] + bank_features].copy()
    
    # Create Insurance DataFrame (with target)
    insurance_df = df[['customer_id'] + insurance_features + ['churn']].copy()
    
    # Save to CSV
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bank_file = output_path / 'bank_data.csv'
    insurance_file = output_path / 'insurance_data.csv'
    
    bank_df.to_csv(bank_file, index=False)
    insurance_df.to_csv(insurance_file, index=False)
    
    print(f"[OK] Converted Telco Churn to Vertical FL format (fallback):")
    print(f"  - Bank data: {len(bank_df)} samples, {len(bank_features)} features")
    print(f"  - Insurance data: {len(insurance_df)} samples, {len(insurance_features)} features")
    print(f"  - Saved to: {bank_file} and {insurance_file}")
    
    return bank_df, insurance_df


def generate_vertical_fl_data(
    n_samples: int = 10000,
    random_state: int = 42,
    save_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data cho Vertical FL.
    
    Tạo 2 datasets:
    1. bank_data.csv: Financial features từ Bank
    2. insurance_data.csv: Policy features từ Insurance
    
    Cả 2 có cùng customer_id để match.
    
    Args:
        n_samples: Số lượng customers
        random_state: Random seed
        save_dir: Directory để save files
    
    Returns:
        Tuple (bank_df, insurance_df)
    """
    np.random.seed(random_state)
    
    # Generate customer IDs
    customer_ids = [f'C{i:06d}' for i in range(1, n_samples + 1)]
    
    # ===== BANK DATA (Financial Features) =====
    bank_data = {
        'customer_id': customer_ids,
        # Financial features
        'balance': np.random.lognormal(mean=10, sigma=0.5, size=n_samples),  # Số dư TK
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),  # Credit score
        'debt_ratio': np.random.beta(2, 5, n_samples),  # Tỷ lệ nợ/thu nhập
        'monthly_income': np.random.lognormal(mean=10.5, sigma=0.4, size=n_samples),  # Thu nhập hàng tháng
        'savings_rate': np.random.beta(3, 7, n_samples),  # Tỷ lệ tiết kiệm
        'transaction_frequency': np.random.poisson(lam=15, size=n_samples),  # Tần suất giao dịch
        'overdraft_count': np.random.poisson(lam=0.5, size=n_samples),  # Số lần thấu chi
    }
    bank_df = pd.DataFrame(bank_data)
    
    # ===== INSURANCE DATA (Policy Features) =====
    # Generate churn probability dựa trên bank features (để có correlation)
    # Customers với credit_score thấp hoặc debt_ratio cao → higher churn probability
    credit_score_norm = (bank_df['credit_score'] - 300) / (850 - 300)
    churn_prob = 0.2 + 0.3 * (1 - credit_score_norm) + 0.2 * bank_df['debt_ratio']
    churn_prob = np.clip(churn_prob, 0.1, 0.9)
    churn = np.random.binomial(1, churn_prob, size=n_samples)
    
    insurance_data = {
        'customer_id': customer_ids,
        # Policy features
        'premium': np.random.lognormal(mean=6, sigma=0.5, size=n_samples),  # Phí bảo hiểm
        'term': np.random.choice([12, 24, 36, 60], size=n_samples, p=[0.3, 0.3, 0.25, 0.15]),  # Kỳ hạn (tháng)
        'coverage': np.random.choice([100000, 200000, 500000, 1000000], size=n_samples, p=[0.2, 0.3, 0.3, 0.2]),  # Mức bảo hiểm
        'age': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),  # Tuổi
        'tenure': np.random.exponential(scale=24, size=n_samples).clip(0, 120).astype(int),  # Thời gian là khách hàng (tháng)
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Cash'], size=n_samples, p=[0.5, 0.3, 0.2]),  # Phương thức thanh toán
        'claims_history': np.random.poisson(lam=0.3, size=n_samples),  # Lịch sử khiếu nại
        'churn': churn,  # Target variable
    }
    insurance_df = pd.DataFrame(insurance_data)
    
    # Save to CSV
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bank_file = output_path / 'bank_data.csv'
    insurance_file = output_path / 'insurance_data.csv'
    
    bank_df.to_csv(bank_file, index=False)
    insurance_df.to_csv(insurance_file, index=False)
    
    print(f"[OK] Generated Vertical FL data:")
    print(f"  - Bank data: {len(bank_df)} samples, {len(bank_df.columns)-1} features")
    print(f"  - Insurance data: {len(insurance_df)} samples, {len(insurance_df.columns)-2} features")
    print(f"  - Saved to: {bank_file} and {insurance_file}")
    
    return bank_df, insurance_df


def load_vertical_fl_data(
    bank_file: Optional[str] = None,
    insurance_file: Optional[str] = None,
    data_dir: str = 'data/raw',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Vertical FL data từ CSV files.
    
    Args:
        bank_file: Path to bank_data.csv
        insurance_file: Path to insurance_data.csv
        data_dir: Directory to search for files
    
    Returns:
        Tuple (bank_df, insurance_df)
    """
    data_path = Path(data_dir)
    
    if bank_file is None:
        bank_file = data_path / 'bank_data.csv'
    else:
        bank_file = Path(bank_file)
    
    if insurance_file is None:
        insurance_file = data_path / 'insurance_data.csv'
    else:
        insurance_file = Path(insurance_file)
    
    if not bank_file.exists():
        raise FileNotFoundError(f"Bank data file not found: {bank_file}")
    if not insurance_file.exists():
        raise FileNotFoundError(f"Insurance data file not found: {insurance_file}")
    
    bank_df = pd.read_csv(bank_file)
    insurance_df = pd.read_csv(insurance_file)
    
    print(f"[OK] Loaded Vertical FL data:")
    print(f"  - Bank data: {len(bank_df)} samples")
    print(f"  - Insurance data: {len(insurance_df)} samples")
    
    return bank_df, insurance_df


def preprocess_vertical_fl_data(
    bank_df: pd.DataFrame,
    insurance_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """Preprocess Vertical FL data.
    
    Args:
        bank_df: Bank DataFrame với customer_id
        insurance_df: Insurance DataFrame với customer_id và target 'churn'
        test_size: Tỷ lệ test set
        random_state: Random seed
    
    Returns:
        Dictionary với:
        - X_bank_train, X_bank_test: Bank features
        - X_insurance_train, X_insurance_test: Insurance features
        - y_train, y_test: Targets
        - customer_ids_train, customer_ids_test: Customer IDs
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Merge để đảm bảo cùng customers
    merged = pd.merge(bank_df, insurance_df, on='customer_id', how='inner')
    
    # Extract features và target
    bank_features = [col for col in bank_df.columns if col != 'customer_id']
    insurance_features = [col for col in insurance_df.columns if col not in ['customer_id', 'churn']]
    
    # Prepare bank features (numerical)
    X_bank = merged[bank_features].copy()
    
    # Handle categorical features in bank data
    for col in bank_features:
        if X_bank[col].dtype == 'object':
            le = LabelEncoder()
            X_bank[col] = le.fit_transform(X_bank[col].astype(str))
    
    X_bank = X_bank.values.astype(float)
    
    # Prepare insurance features (categorical + numerical)
    X_insurance = merged[insurance_features].copy()
    
    # Encode categorical features trong insurance data
    le_dict = {}
    for col in insurance_features:
        if X_insurance[col].dtype == 'object':
            le = LabelEncoder()
            X_insurance[col] = le.fit_transform(X_insurance[col].astype(str))
            le_dict[col] = le
    
    X_insurance = X_insurance.values.astype(float)
    
    y = merged['churn'].values
    customer_ids = merged['customer_id'].values
    
    # Train/test split
    indices = np.arange(len(X_bank))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    X_bank_train = X_bank[train_idx]
    X_bank_test = X_bank[test_idx]
    X_insurance_train = X_insurance[train_idx]
    X_insurance_test = X_insurance[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    customer_ids_train = customer_ids[train_idx]
    customer_ids_test = customer_ids[test_idx]
    
    # Scale features
    scaler_bank = StandardScaler()
    scaler_insurance = StandardScaler()
    
    X_bank_train = scaler_bank.fit_transform(X_bank_train)
    X_bank_test = scaler_bank.transform(X_bank_test)
    
    X_insurance_train = scaler_insurance.fit_transform(X_insurance_train)
    X_insurance_test = scaler_insurance.transform(X_insurance_test)
    
    return {
        'X_bank_train': X_bank_train,
        'X_bank_test': X_bank_test,
        'X_insurance_train': X_insurance_train,
        'X_insurance_test': X_insurance_test,
        'y_train': y_train,
        'y_test': y_test,
        'customer_ids_train': customer_ids_train,
        'customer_ids_test': customer_ids_test,
        'bank_features': bank_features,
        'insurance_features': insurance_features,
        'scaler_bank': scaler_bank,
        'scaler_insurance': scaler_insurance,
        'label_encoders': le_dict,
    }
