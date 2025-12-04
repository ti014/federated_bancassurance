# Data Directory

Đặt các file dataset CSV vào đây.

## Khuyến Nghị Datasets

1. **Bank Customer Churn** (Khuyến nghị nhất)
   - Download từ: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
   - File name: `Bank Customer Churn Prediction.csv` hoặc `bank_churn.csv`


## Cách Sử Dụng

Sau khi đặt file CSV vào đây, code sẽ tự động tìm và load:

```python
from data.loader import load_dataset
features, target = load_dataset()  # Tự động tìm file
```

Xem `DATASETS_GUIDE.md` để biết chi tiết.

