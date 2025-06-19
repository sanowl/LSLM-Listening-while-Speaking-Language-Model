# LSLM Project Structure

```
lslm/
├── src/                      # Source code
│   └── lslm/
│       ├── models/          # Model implementations
│       │   ├── __init__.py
│       │   ├── lslm.py      # Main LSLM model
│       │   └── layers/      # Custom model layers
│       ├── data/            # Data handling
│       │   ├── __init__.py
│       │   ├── dataset.py   # Dataset classes
│       │   └── processors/  # Data processors
│       ├── utils/           # Utilities
│       │   ├── __init__.py
│       │   ├── training.py  # Training utilities
│       │   └── metrics.py   # Evaluation metrics
│       ├── configs/         # Configuration
│       │   ├── __init__.py
│       │   └── default_config.py
│       └── tokenizers/      # Custom tokenizers
│           └── __init__.py
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   │   ├── test_model.py
│   │   ├── test_dataset.py
│   │   ���── test_training.py
│   └── integration/        # Integration tests
│       └── test_pipeline.py
├── scripts/                # Utility scripts
│   ├── train_aws.py       # AWS training
│   ├── preprocess_data.py # Data preprocessing
│   ├── evaluate.py        # Model evaluation
│   └── export_model.py    # Model export
├── configs/               # Configuration files
│   ├── model/            # Model configs
│   │   ├── base.yaml
│   │   └── large.yaml
│   └── training/         # Training configs
│       ├── default.yaml
│       └── distributed.yaml
├── docs/                 # Documentation
│   ├── api/             # API documentation
│   ├── examples/        # Usage examples
│   └── tutorials/       # Step-by-step guides
├── examples/            # Example code
│   ├── inference.py
│   └── streaming.py
├── notebooks/          # Jupyter notebooks
│   ├── exploration.ipynb
│   └── demo.ipynb
├── requirements.txt    # Project dependencies
├── setup.py           # Package setup
├── .gitignore         # Git ignore rules
├── README.md          # Project overview
└── LICENSE           # License information
```

## Directory Descriptions

### Source Code (`src/lslm/`)
- `models/`: Model architectures and layers
- `data/`: Data loading and processing
- `utils/`: Helper functions and utilities
- `configs/`: Configuration management
- `tokenizers/`: Custom tokenization logic

### Tests (`tests/`)
- `unit/`: Individual component tests
- `integration/`: End-to-end tests

### Scripts (`scripts/`)
- Utility scripts for training, preprocessing, etc.
- AWS deployment scripts

### Configs (`configs/`)
- YAML configuration files
- Separate model and training configs

### Documentation (`docs/`)
- API documentation
- Usage examples
- Tutorials

### Examples (`examples/`)
- Example implementations
- Demo applications

### Notebooks (`notebooks/`)
- Exploratory analysis
- Interactive demos 