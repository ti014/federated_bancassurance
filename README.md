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
git clone https://github.com/ti014/federated_bancassurance.git
cd federated_bancassurance
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
│   ├── generate_roc_curve.py      # Generate ROC curve 
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
│   │   ├── roc_curve.png          # ROC Curve với AUC 
│   │   └── ...                    # Other plots
│   ├── privacy_analysis/          # Privacy analysis results 
│   │   ├── privacy_analysis_results.csv
│   │   └── privacy_analysis_plots.png
│   └── logs/                      # Experiment logs
│       ├── training.log           # Training logs
│       ├── privacy_analysis.log   # Privacy analysis logs 
│       └── roc_curve.log          # ROC curve logs 
├── EXPERIMENTS_SUMMARY.md         # Tổng hợp kết quả experiments 
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
- **ROC-AUC:** 0.8437 (84.37%) 
- **Accuracy:** 0.8500 (85.00%) 
- **Precision:** 0.6450 (64.50%) 
- **Recall:** 0.5848 (58.48%)
- **F1-Score:** 0.6134 (61.34%) 
- **Optimal Threshold:** 0.71 (tự động tìm để maximize F1)

**Model Configuration:**
- Embedding size: 64 (tăng từ 32)
- Bottom hidden sizes: [256, 128, 64]
- Top hidden sizes: [128, 64, 32]
- Dropout: 0.2
- Batch size: 64
- Learning rate: 0.0005 (với ReduceLROnPlateau)
- Epochs: 100 (với early stopping)
