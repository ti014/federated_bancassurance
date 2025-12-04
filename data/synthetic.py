"""Synthetic data generator for Life Insurance Lapse Prediction.

Tạo dữ liệu giả lập cho bài toán Lapse Prediction trong Bancassurance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


def generate_synthetic_insurance_data(
    n_samples: int = 10000,
    n_banks: int = 5,
    random_state: Optional[int] = 42,
    non_iid_ratio: float = 0.3,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic insurance dataset for Lapse Prediction.
    
    Args:
        n_samples: Tổng số samples cần generate
        n_banks: Số lượng ngân hàng (clients) để simulate
        random_state: Random seed để reproducibility
        non_iid_ratio: Tỷ lệ Non-IID distribution (0.0 = IID, 1.0 = highly Non-IID)
    
    Returns:
        Tuple của (features DataFrame, target Series)
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {}
    
    # Customer age (18-70)
    data['customer_age'] = np.random.normal(45, 15, n_samples).clip(18, 70)
    
    # Premium amount (log-normal distribution)
    data['premium_amount'] = np.random.lognormal(mean=10, sigma=0.8, size=n_samples).clip(1000, 500000)
    
    # Policy term (years)
    data['policy_term'] = np.random.choice([5, 10, 15, 20, 25, 30], size=n_samples, p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05])
    
    # Annual income (correlated with premium)
    data['annual_income'] = data['premium_amount'] * np.random.uniform(8, 15, n_samples)
    
    # Number of dependents
    data['num_dependents'] = np.random.poisson(2, n_samples).clip(0, 5)
    
    # Employment status (encoded: 0=Unemployed, 1=Employed, 2=Self-employed)
    data['employment_status'] = np.random.choice([0, 1, 2], size=n_samples, p=[0.1, 0.7, 0.2])
    
    # Marital status (0=Single, 1=Married)
    data['marital_status'] = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    
    # Bank ID (categorical, sẽ dùng để split clients)
    data['bank_id'] = np.random.choice(range(n_banks), size=n_samples)
    
    # Health status score (0-100, higher = better health)
    data['health_score'] = np.random.normal(70, 20, n_samples).clip(0, 100)
    
    # Years since policy start
    data['years_since_start'] = np.random.exponential(5, n_samples).clip(0, 30)
    
    df = pd.DataFrame(data)
    
    # Generate target (lapse) với correlation với features
    # Higher lapse probability nếu:
    # - Premium quá cao so với income
    # - Age cao
    # - Health score thấp
    # - Policy term dài
    
    premium_income_ratio = df['premium_amount'] / df['annual_income']
    age_factor = (df['customer_age'] - 45) / 20  # Normalize around 45
    health_factor = (70 - df['health_score']) / 70  # Inverse health
    
    # Base probability
    lapse_prob = (
        0.15  # Base rate
        + 0.3 * np.clip(premium_income_ratio - 0.1, 0, 0.2) / 0.2  # Financial stress
        + 0.2 * np.clip(age_factor, 0, 1)  # Age factor
        + 0.15 * health_factor  # Health factor
        + 0.1 * (df['policy_term'] > 20).astype(float)  # Long term
        + np.random.normal(0, 0.1, n_samples)  # Noise
    )
    
    # Apply Non-IID: một số banks có tỷ lệ lapse cao hơn
    if non_iid_ratio > 0:
        high_lapse_banks = np.random.choice(n_banks, size=int(n_banks * non_iid_ratio), replace=False)
        for bank_id in high_lapse_banks:
            mask = df['bank_id'] == bank_id
            lapse_prob[mask] += 0.2  # Increase lapse probability
    
    lapse_prob = np.clip(lapse_prob, 0, 1)
    target = (np.random.random(n_samples) < lapse_prob).astype(int)
    
    # Drop bank_id từ features (sẽ dùng riêng để split)
    features = df.drop('bank_id', axis=1)
    
    return features, pd.Series(target, name='lapse')


def generate_bank_specific_data(
    bank_id: int,
    n_samples: int,
    base_features: Optional[pd.DataFrame] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate data cho một bank cụ thể (dùng trong FL simulation).
    
    Args:
        bank_id: ID của bank
        n_samples: Số samples cho bank này
        base_features: DataFrame features mẫu (optional)
        random_state: Random seed
    
    Returns:
        Tuple của (features DataFrame, target Series)
    """
    if random_state is not None:
        np.random.seed(random_state + bank_id)
    
    # Generate features tương tự nhưng có thể có distribution khác nhau
    features, target = generate_synthetic_insurance_data(
        n_samples=n_samples,
        n_banks=1,  # Không cần bank_id trong output
        random_state=random_state + bank_id if random_state else None,
    )
    
    return features, target

