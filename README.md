```
fraud-anomaly-detection/
│
├── data/
│   ├── raw/
│   │   └── transactions_raw.csv
│   │
│   ├── processed/
│   │   └── transactions_features.csv
│   │
│   └── metadata/
│       ├── city_coordinates.json
│       └── merchant_categories.json
│
├── src/
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── city_loader.py
│   │   ├── customer_generator.py
│   │   ├── merchant_generator.py
│   │   ├── transaction_simulator.py
│   │   ├── fraud_rules.py
│   │   └── generate_dataset.py
│   │
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── temporal_features.py
│   │   ├── spatial_features.py
│   │   ├── behavioral_features.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── isolation_forest.py
│   │   ├── one_class_svm.py
│   │   ├── autoencoder.py
│   │   └── model_utils.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── threshold_tuning.py
│   │   └── edge_case_analysis.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── inference.py
│   │   └── logger.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── geo_utils.py
│       ├── time_utils.py
│       └── config.py
│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_threshold_analysis.ipynb
│
├── reports/
│   ├── figures/
│   └── fraud_detection_technical_report.pdf
│
├── tests/
│   ├── test_fraud_rules.py
│   ├── test_distance_calculation.py
│   └── test_api.py
│
├── requirements.txt
├── README.md
└── run_pipeline.py
```