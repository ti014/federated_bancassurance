# Vertical Federated Learning với SplitNN cho Lapse Prediction trong Bancassurance

Đồ án nghiên cứu và triển khai **Vertical Federated Learning (VFL)** với **Split Neural Network (SplitNN)** để dự đoán **Lapse (để mất hiệu lực hợp đồng)** trong kênh Bancassurance.

## Mục Tiêu

- Hiểu và trình bày lý thuyết **Vertical Federated Learning** và **SplitNN**
- Implement VFL framework với SplitNN architecture:
  - **Bottom Model** (Bank): Xử lý financial features → embedding `h_B`
  - **Top Model** (Insurance): Nhận `h_B` + policy features → prediction
- So sánh VFL vs Centralized Learning
- Đánh giá **Privacy-preserving**: Chứng minh không thể recover raw data từ embedding

## Tech Stack

- **Python 3.10+**
- **PyTorch** (1.13+): Deep Learning framework
- **Pandas + NumPy**: Data manipulation
- **Scikit-learn**: Preprocessing và metrics
- **Matplotlib + Seaborn**: Visualization
- **imbalanced-learn**: SMOTE cho class imbalance handling

## Cài Đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd federated_bancassurance_report
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

**Khuyến nghị: Bank Customer Churn** (phù hợp nhất với Bancassurance domain)

1. Download từ Kaggle: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
2. Đặt file vào `data/raw/` với tên `bank_churn.csv` hoặc `Bank Customer Churn Prediction.csv`

Nếu không có dataset, code sẽ tự động generate synthetic data.

## Cấu Trúc Project

```
federated_bancassurance_report/
├── data/                          # Data Module
│   ├── __init__.py                # Export: vertical_split, datasets functions
│   ├── datasets.py                # Load public datasets (Bank Churn)
│   ├── vertical_split.py          # Split data cho Vertical FL (Bank vs Insurance)
│   ├── loader.py                  # Legacy: Horizontal FL data loader
│   └── synthetic.py               # Generate synthetic data (fallback)
│   └── raw/                       # Dataset files (CSV)
│       ├── README.md              # Hướng dẫn download datasets
│       └── bank_churn.csv         # Bank Customer Churn dataset
├── models/                        # Models Module
│   ├── __init__.py                # Export: SplitNN models
│   ├── splitnn.py                 # BottomModel, TopModel, create_splitnn_models
│   └── baseline.py                # Baseline models (Logistic Regression, Random Forest)
├── federated/                     # Federated Learning Module
│   ├── __init__.py                # Export: Clients và Server
│   ├── vertical_client.py         # BankBottomClient, InsuranceTopClient
│   └── vertical_server.py          # VerticalFLServer (Coordinator)
├── experiments/                   # Experiments Module
│   ├── __init__.py                # Export: Training functions
│   ├── vertical_fl.py             # run_vertical_fl_training() - MAIN
│   ├── vertical_fl_with_server.py # Version với Server (alternative)
│   ├── centralized.py             # Centralized baseline
│   └── compare_vfl_centralized.py # Comparison script
├── utils/                         # Utilities Module
│   ├── __init__.py                # Export: metrics, visualization, config, logger
│   ├── metrics.py                 # Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
│   ├── visualization.py           # Plotting functions
│   ├── config.py                  # Config loader (YAML)
│   └── logger.py                  # Logging utility
├── scripts/                       # Scripts Module
│   ├── run_vertical_fl.py         # MAIN ENTRY POINT: Chạy Vertical FL training
│   ├── evaluate_training_results.py # Đánh giá kết quả training
│   ├── show_results.py            # Hiển thị kết quả cuối cùng
│   ├── load_model.py              # Load trained models
│   ├── inference.py               # Inference với trained models
│   ├── run_privacy_analysis.py     # Privacy analysis (wrapper)
│   ├── generate_roc_curve.py      # Generate ROC curve ✅
│   ├── find_optimal_threshold.py  # Tìm optimal threshold cho F1
│   └── test_optimal_threshold.py  # Test với các thresholds khác nhau
├── tests/                         # Unit Tests
│   ├── __init__.py
│   ├── test_models.py             # Tests cho models module
│   ├── test_data.py               # Tests cho data module
│   └── test_federated.py          # Tests cho federated module
├── configs/                       # Configuration Files
│   ├── model_config.yaml          # Model hyperparameters
│   └── fl_config.yaml             # FL hyperparameters
├── results/                       # Results Output
│   ├── models/                    # Saved model checkpoints
│   │   ├── bank_bottom_model.pth  # Bottom Model (Bank side) - Vertical FL
│   │   └── insurer_top_model.pth  # Top Model (Insurance side) - Vertical FL
│   ├── plots/                     # Generated visualizations
│   │   ├── roc_curve.png          # ROC Curve với AUC ✅
│   │   └── ...                    # Other plots
│   ├── privacy_analysis/          # Privacy analysis results ✅
│   │   ├── privacy_analysis_results.csv
│   │   └── privacy_analysis_plots.png
│   └── logs/                      # Experiment logs
│       ├── training.log           # Training logs
│       ├── privacy_analysis.log   # Privacy analysis logs ✅
│       └── roc_curve.log          # ROC curve logs ✅
├── EXPERIMENTS_SUMMARY.md         # Tổng hợp kết quả experiments ✅
├── run_tests.py                   # Script để chạy unit tests
├── setup.py                       # Package setup
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Kiến Trúc Vertical FL

### SplitNN Architecture với Server Coordinator

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   Bank Client   │         │  FL Server       │         │ Insurance Client │
│                 │         │  (Coordinator)   │         │                  │
│  X_B (financial)│         │                  │         │  X_I (policy)    │
│       ↓         │         │                  │         │                  │
│  Bottom Model   │         │                  │         │                  │
│       ↓         │         │                  │         │                  │
│   h_B (embed)   │────────>│  Forward:        │────────>│  Top Model       │
│                 │         │  h_B → Insurance │         │  [h_B, X_I]      │
│                 │<────────│  Backward:       │<────────│       ↓          │
│                 │         │  grad → Bank     │         │   y_pred         │
└─────────────────┘         └──────────────────┘         └──────────────────┘
```

### Vai Trò của Server

**Server đóng vai trò Coordinator:**
- Điều phối forward pass: Nhận embedding `h_B` từ Bank → Gửi cho Insurance
- Điều phối backward pass: Nhận gradients từ Insurance → Gửi về Bank
- Quản lý training rounds và synchronization
- Đảm bảo các clients không trao đổi raw data trực tiếp

### Data Flow

1. **Bank**: Financial features → Bottom Model → Embedding `h_B`
2. **Bank → Server**: Gửi `h_B` cho Server (không gửi raw financial data)
3. **Server → Insurance**: Server forward `h_B` cho Insurance
4. **Insurance**: Nhận `h_B` + Policy features → Top Model → Prediction `y_pred`
5. **Insurance → Server**: Tính loss và gradients, gửi gradients về Server
6. **Server → Bank**: Server forward gradients về Bank
7. **Bank**: Update Bottom Model với gradients từ Server
8. **Insurance**: Update Top Model với gradients của chính nó

## Sử Dụng

### Quick Start - Vertical FL (Khuyến Nghị)

**Chạy Vertical FL training:**

```bash
python scripts/run_vertical_fl.py
```

Script này sẽ tự động:
1. Load Bank Customer Churn dataset
2. Convert thành Vertical FL format (Bank features + Insurance features)
3. Preprocess và split data (80% train, 20% test)
4. Train với SplitNN architecture
5. Apply SMOTE để xử lý class imbalance
6. Save models: `bank_bottom_model.pth` và `insurer_top_model.pth`
7. Log tất cả thông tin vào `results/logs/training.log`

**Kết quả mẫu (với Bank Customer Churn - Improved Model):**
- **ROC-AUC:** 0.8437 (84.37%) ✅
- **Accuracy:** 0.8500 (85.00%) ✅
- **Precision:** 0.6450 (64.50%) ✅
- **Recall:** 0.5848 (58.48%)
- **F1-Score:** 0.6134 (61.34%) ✅
- **Optimal Threshold:** 0.71 (tự động tìm để maximize F1)

**Model Configuration:**
- Embedding size: 64 (tăng từ 32)
- Bottom hidden sizes: [256, 128, 64]
- Top hidden sizes: [128, 64, 32]
- Dropout: 0.2
- Batch size: 64
- Learning rate: 0.0005 (với ReduceLROnPlateau)
- Epochs: 100 (với early stopping)

### Đánh Giá Kết Quả

```bash
python scripts/evaluate_training_results.py
```

### Xem Kết Quả Cuối Cùng

```bash
python scripts/show_results.py
```

### Chạy Unit Tests

```bash
python run_tests.py
```

## Dataset Details

### Bank Customer Churn Dataset

**Bank Features (10 features):**
- `CreditScore`: Credit score của khách hàng
- `Geography`: Địa lý (France, Spain, Germany)
- `Gender`: Giới tính
- `Age`: Tuổi
- `Tenure`: Thời gian là khách hàng (năm)
- `Balance`: Số dư tài khoản
- `NumOfProducts`: Số sản phẩm ngân hàng đang dùng
- `HasCrCard`: Có thẻ tín dụng
- `IsActiveMember`: Là thành viên tích cực
- `EstimatedSalary`: Thu nhập ước tính

**Insurance Features (6 features - được tạo từ Bank context):**
- `premium`: Phí bảo hiểm hàng tháng (tính từ Balance và Salary)
- `policy_term`: Kỳ hạn hợp đồng (năm) - tính từ Tenure
- `coverage`: Mức bảo hiểm (tính từ Salary)
- `payment_frequency`: Tần suất thanh toán (Monthly/Quarterly)
- `policy_type`: Loại bảo hiểm (Basic/Premium)
- `age_group`: Nhóm tuổi

**Target:** `Exited` → `lapse` (1 = Lapse, 0 = No Lapse)

## Model Architecture

### Bottom Model (Bank)

- Input: 10 Bank features
- Hidden layers: **[256, 128, 64]** với ReLU activation (tăng capacity)
- Output: Embedding vector **(64 dimensions)** (tăng từ 32)
- Dropout: 0.2

### Top Model (Insurance)

- Input: Embedding (64) + 6 Insurance features = **70 dimensions**
- Hidden layers: **[128, 64, 32]** với ReLU activation (tăng capacity)
- Output: 1 (probability of lapse)
- Dropout: 0.2

### Training Configuration

- Batch size: 64
- Learning rate: 0.0005 (với ReduceLROnPlateau scheduler)
- Max epochs: 100 (với early stopping, patience=20)
- Weight decay: 1e-5 (L2 regularization)
- Loss: BCEWithLogitsLoss với pos_weight (handle class imbalance)
- SMOTE: Enabled để oversample minority class

## Sử Dụng Trained Models

### 1. Load Models

```bash
# Load Bottom Model (Bank)
python scripts/load_model.py --bottom-checkpoint results/models/bank_bottom_model.pth --top-checkpoint results/models/insurer_top_model.pth

# Load Top Model (Insurance)
python scripts/load_model.py --top-checkpoint results/models/insurer_top_model.pth
```

### 2. Inference với Vertical FL Models

Để inference với Vertical FL, cần cả 2 models:

```python
import torch
from models.splitnn import BottomModel, TopModel

# Load models
bottom_model = BottomModel(input_size=10, embedding_size=64, hidden_sizes=[256, 128, 64])
top_model = TopModel(embedding_size=64, insurance_input_size=6, hidden_sizes=[128, 64, 32])

bottom_model.load_state_dict(torch.load('results/models/bank_bottom_model.pth')['model_state_dict'])
top_model.load_state_dict(torch.load('results/models/insurer_top_model.pth')['model_state_dict'])

bottom_model.eval()
top_model.eval()

# Inference
bank_features = torch.FloatTensor([[619, 0, 1, 42, 2, 0, 1, 1, 1, 101348.88]])  # Example
insurance_features = torch.FloatTensor([[500, 1, 1000000, 1, 1, 1]])  # Example

with torch.no_grad():
    embedding = bottom_model(bank_features)
    prediction = top_model(embedding, insurance_features)
    probability = torch.sigmoid(prediction)
    
print(f"Lapse probability: {probability.item():.4f}")
```

### 3. Privacy Analysis

```bash
python scripts/run_privacy_analysis.py
```

Script này sẽ phân tích khả năng recover raw data từ embedding bằng linear regression. Kết quả cho thấy mức độ privacy protection của embedding.

**Kết quả mẫu:**
- Average R²: 0.8359 (HIGH reconstruction quality)
- Privacy Level: LOW
- Recommendation: Cần thêm privacy mechanisms (noise, differential privacy)

### 4. Generate ROC Curve

```bash
python scripts/generate_roc_curve.py
```

Generate ROC curve và tính AUC score cho trained model.

**Kết quả:**
- ROC Curve: `results/plots/roc_curve.png`
- AUC Score: ~0.84

## Kết Quả

Sau khi chạy `scripts/run_vertical_fl.py`, kết quả sẽ được lưu trong `results/`:

- `results/models/`:
  - `bank_bottom_model.pth`: Bottom Model (Bank side)
  - `insurer_top_model.pth`: Top Model (Insurance side)
- `results/plots/`:
  - `vfl_loss_curves.png`: Loss curves (train/test)
  - `vfl_accuracy_curves.png`: Accuracy curves (train/test)
  - `vfl_train_test_loss.png`: Train vs Test Loss comparison
  - `vfl_train_test_accuracy.png`: Train vs Test Accuracy comparison
  - `vfl_metrics_bar.png`: Final metrics bar chart
  - `roc_curve.png`: ROC Curve với AUC score ✅
  - `vfl_loss_comparison.png`: VFL vs Centralized Loss comparison (nếu có)
  - `vfl_accuracy_comparison.png`: VFL vs Centralized Accuracy comparison (nếu có)
  - `vfl_metrics_comparison.png`: VFL vs Centralized Metrics comparison (nếu có)
- `results/privacy_analysis/`:
  - `privacy_analysis_results.csv`: Privacy analysis results
  - `privacy_analysis_plots.png`: Recovery quality plots
- `results/logs/`:
  - `training.log`: Training logs với timestamp và level
  - `privacy_analysis.log`: Privacy analysis logs
  - `roc_curve.log`: ROC curve generation logs

## Experiments

### Experiment 1: VFL vs Centralized ⏳

So sánh accuracy và convergence speed giữa VFL và Centralized Learning.

**Chạy:**
```bash
python experiments/compare_vfl_centralized.py
```

Kết quả được lưu tại:
- `results/comparison_vfl_centralized.txt`: Chi tiết comparison
- `results/plots/vfl_loss_comparison.png`: Loss comparison plot
- `results/plots/vfl_accuracy_comparison.png`: Accuracy comparison plot
- `results/plots/vfl_metrics_comparison.png`: Metrics comparison bar chart

### Experiment 2: Class Imbalance Handling ✅

Đánh giá hiệu quả của SMOTE và weighted loss trong việc xử lý class imbalance.

**Kết quả:**
- SMOTE được áp dụng tự động trong training
- Weighted loss (pos_weight) được tính từ training data
- F1 Score: ~61.34% (cải thiện từ baseline)

### Experiment 3: Privacy Analysis ✅

Phân tích khả năng recover raw data từ embedding `h_B` bằng linear regression.

**Kết quả:**
- Average R² (reconstruction quality): **0.8359**
- Privacy Level: **LOW** ⚠️
- **Kết luận:** Embedding có thể leak information. Đề xuất thêm differential privacy hoặc noise để cải thiện privacy.

**Chạy:**
```bash
python scripts/run_privacy_analysis.py
```

Kết quả được lưu tại `results/privacy_analysis/`:
- `privacy_analysis_results.csv`: Chi tiết recovery quality cho từng feature
- `privacy_analysis_plots.png`: Visualization recovery quality

### Experiment 4: ROC Curve Analysis ✅

Generate ROC curve và tính AUC score để đánh giá model performance.

**Chạy:**
```bash
python scripts/generate_roc_curve.py
```

**Kết quả:**
- ROC Curve: `results/plots/roc_curve.png`
- AUC Score: **0.8437** (84.37%)

**Tổng hợp kết quả:** Xem `EXPERIMENTS_SUMMARY.md` để biết chi tiết tất cả experiments.

## Lưu Ý

- Code được thiết kế để reproducible: tất cả random seeds được set
- Ưu tiên sử dụng Bank Customer Churn dataset (phù hợp với Bancassurance domain)
- Nếu không có dataset, code sẽ tự động generate synthetic data
- Model có early stopping để tránh overfitting
- SMOTE được áp dụng tự động để xử lý class imbalance
- Logging được ghi vào file `results/logs/training.log` và console
- Unit tests có sẵn trong `tests/` directory

## Tài Liệu Tham Khảo

- [Flower Documentation](https://flower.dev/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Federated Learning Papers](https://arxiv.org/search/?query=federated+learning)
- [Vertical Federated Learning Survey](https://arxiv.org/abs/2101.09426)

## License

MIT License

## Tác Giả

Đồ án Machine Learning - Federated Learning cho Lapse Prediction trong Bancassurance Channel
