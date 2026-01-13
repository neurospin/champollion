# Config Manager Module

This module provides a decoupled configuration management system for Champollion V1, separating dataset definitions from code to improve maintainability.

## 📦 Components

### 1. **DatasetRegistry** (`dataset_registry.py`)
Manages dataset configurations dynamically without hardcoding paths.

```python
from contrastive.config_manager import DatasetRegistry

# Auto-discover all datasets in configs/dataset directory
registry = DatasetRegistry(config_dir="./configs/dataset", auto_discover=True)

# List all datasets
datasets = registry.list_datasets()

# Get specific dataset
dataset = registry.get("gridsearch_2024/hcp_left")

# Filter datasets by pattern
hcp_datasets = registry.filter_datasets("*/hcp*")
```

### 2. **ConfigLoader** (`config_loader.py`)
Loads configurations from external YAML files.

```python
from contrastive.config_manager import ConfigLoader

loader = ConfigLoader()

# Load datasets by name
datasets_dict = loader.load_datasets_from_names(
    dataset_names=["gridsearch_2024/hcp_left", "gridsearch_2024/hcp_right"],
    datasets_root=None
)

# Load label configuration
label_config = loader.load_label_config("Sex")

# Load classifier configuration
classifier_config = loader.load_classifier_config("svm")

# Create complete evaluation config
config = loader.create_evaluation_config(
    model_path="/path/to/model",
    dataset_names=["my_dataset"],
    label_name="Sex",
    dataset_localization="local",
    classifier_name="svm"
)
```

### 3. **HydraBridge** (`hydra_bridge.py`)
Provides drop-in replacements for `utils_pipelines.py` functions.

```python
from contrastive.config_manager import HydraBridge

bridge = HydraBridge()

# Replace datasets in config (instead of hardcoded yaml.load)
bridge.change_config_datasets(config, ["dataset1", "dataset2"], datasets_root=None)

# Change label
bridge.change_config_label(config, "Age")

# Change dataset localization
bridge.change_config_dataset_localization(config, "local")

# List available resources
datasets = bridge.list_available_datasets(pattern="gridsearch_2024/*")
labels = bridge.list_available_labels()
classifiers = bridge.list_available_classifiers()
```

## 🔧 Integration with Existing Code

### Option 1: Use `utils_pipelines_new.py` (Recommended)

This provides drop-in replacements that work exactly like the original functions:

```python
# OLD CODE (utils_pipelines.py)
from contrastive.evaluation.utils_pipelines import (
    change_config_datasets,
    change_config_label,
    get_save_folder_name
)

# NEW CODE (utils_pipelines_new.py)
from contrastive.evaluation.utils_pipelines_new import (
    change_config_datasets,  # Now uses DatasetRegistry
    change_config_label,     # Now uses ConfigLoader
    get_save_folder_name
)

# Usage is IDENTICAL - just import from different module
change_config_datasets(config, ["dataset1"], None)
change_config_label(config, "Sex")
```

### Option 2: Update `embeddings_pipeline.py`

Replace the imports at the top of [embeddings_pipeline.py](../embeddings_pipeline.py):

```python
# Change this:
from utils_pipelines import get_save_folder_name, change_config_datasets, \
                           change_config_label, change_config_dataset_localization

# To this:
from utils_pipelines_new import get_save_folder_name, change_config_datasets, \
                                change_config_label, change_config_dataset_localization
```

That's it! No other code changes needed.

### Option 3: Use HydraBridge Directly

For more control, use HydraBridge directly:

```python
from contrastive.config_manager import HydraBridge

bridge = HydraBridge()

# In preprocess_config function:
def preprocess_config(sub_dir, dataset_localization, datasets_root, datasets, ...):
    cfg = omegaconf.OmegaConf.load(join(sub_dir, '.hydra', 'config.yaml'))

    # Use bridge methods instead of hardcoded yaml.load
    bridge.change_config_datasets(cfg, datasets, datasets_root)
    bridge.change_config_label(cfg, label)
    bridge.change_config_dataset_localization(cfg, dataset_localization)

    return cfg
```

## 🎯 Benefits

### Before (Hardcoded)
```python
# utils_pipelines.py - Line 55
def change_config_datasets(config, new_datasets, new_datasets_root):
    for dataset in new_datasets:
        with open("/volatile/home/bd285800/Documents/CEA_projects/champollion/data/TEST01/derivatives/champollion_V1", 'r') as file:
            dataset_yaml = yaml.load(file, yaml.FullLoader)  # ❌ Hardcoded path!
```

### After (Dynamic)
```python
# utils_pipelines_new.py
def change_config_datasets(config, new_datasets, new_datasets_root):
    bridge = _get_bridge()
    bridge.change_config_datasets(config, new_datasets, new_datasets_root)
    # ✅ Loads from DatasetRegistry - no hardcoded paths!
```

## 📝 Creating New Datasets

### Method 1: Create YAML File Manually

Create a YAML file in `configs/dataset/my_datasets/`:

```yaml
# configs/dataset/my_datasets/my_experiment.yaml
dataset_name: my_experiment
numpy_all: /path/to/skeleton.npy
subjects_all: /path/to/skeleton_subject.csv
foldlabel_all: /path/to/label.npy
subjects_foldlabel_all: /path/to/label_subject.csv
crop_dir: /path/to/crops
crop_file_suffix: _cropped_skeleton.nii.gz
foldlabel_dir: /path/to/labels
train_val_csv_file: /path/to/train_val.csv
subject_labels_file: /path/to/subject_labels.csv
subject_column_name: participant_id
input_size: "(1, 18, 41, 38)"
```

The DatasetRegistry will auto-discover it on next load.

### Method 2: Use generate_champollion_config.py

This script already exists and creates dataset configs:

```bash
python champollion_pipeline/src/generate_champollion_config.py \
    /path/to/crops \
    --dataset my_experiment \
    --champollion_loc ./champollion_V1/
```

### Method 3: Use HydraBridge Programmatically

```python
from contrastive.config_manager import HydraBridge

bridge = HydraBridge()

bridge.create_dataset_from_directory(
    dataset_name="my_experiment",
    base_dir="/path/to/data",
    subject_labels_file="/path/to/labels.csv",
    train_val_csv_file="/path/to/train_val.csv",
    region="left",
    input_size="(1, 18, 41, 38)"
)
```

## 🚀 Usage Examples

### Example 1: Update embeddings_pipeline.py to Use Config Manager

```python
# At top of embeddings_pipeline.py, change:
from utils_pipelines import (
    get_save_folder_name,
    change_config_datasets,
    change_config_label,
    change_config_dataset_localization
)

# To:
from utils_pipelines_new import (
    get_save_folder_name,
    change_config_datasets,
    change_config_label,
    change_config_dataset_localization
)

# Rest of code remains EXACTLY the same!
```

### Example 2: List Available Datasets

```python
from contrastive.config_manager import HydraBridge

bridge = HydraBridge()

# List all datasets
all_datasets = bridge.list_available_datasets(format_output=True)

# Filter by pattern
hcp_datasets = bridge.list_available_datasets(pattern="*/hcp*", format_output=True)

# List labels and classifiers
labels = bridge.list_available_labels(format_output=True)
classifiers = bridge.list_available_classifiers(format_output=True)
```

### Example 3: Create Config for Evaluation

```python
from contrastive.config_manager import ConfigLoader

loader = ConfigLoader()

# Create evaluation config
eval_config = loader.create_evaluation_config(
    model_path="/path/to/trained/model",
    dataset_names=["gridsearch_2024/hcp_left", "gridsearch_2024/hcp_right"],
    label_name="Sex",
    dataset_localization="local",
    classifier_name="svm",
    # Additional overrides
    cv=10,
    split="random",
    overwrite=True
)

# Save for later use
loader.save_config(eval_config, "my_evaluation_config.yaml")
```

## 🔄 Migration Guide

### Step 1: Install Config Manager

The config_manager module is already in your champollion_V1 directory:
```
champollion_V1/contrastive/config_manager/
├── __init__.py
├── dataset_registry.py
├── config_loader.py
├── hydra_bridge.py
└── utils_pipelines_new.py (drop-in replacement)
```

### Step 2: Update Import Statements

In files that use `utils_pipelines.py`, change imports:

```python
# OLD
from contrastive.evaluation.utils_pipelines import change_config_datasets

# NEW
from contrastive.evaluation.utils_pipelines_new import change_config_datasets
```

### Step 3: Test

```python
# Test that datasets are loaded correctly
from contrastive.config_manager import DatasetRegistry

registry = DatasetRegistry(config_dir="./contrastive/configs/dataset")
print(f"Loaded {len(registry.list_datasets())} datasets")
print(registry.list_datasets()[:10])  # Print first 10
```

## 🐛 Troubleshooting

### Issue: "Dataset not found in registry"

**Solution:** Check dataset path and reload registry:

```python
from contrastive.config_manager import ConfigLoader

loader = ConfigLoader()
registry = loader.get_dataset_registry(force_reload=True)
print(registry.list_datasets())
```

### Issue: "Config file not found"

**Solution:** Verify config_root is correct:

```python
from contrastive.config_manager import ConfigLoader

loader = ConfigLoader()
print(f"Config root: {loader.config_root}")
print(f"Dataset dir: {loader.config_root}/dataset")
```

### Issue: Hardcoded path still in utils_pipelines.py

**Solution:** Use `utils_pipelines_new.py` instead:

```python
# Import from new module
from contrastive.evaluation.utils_pipelines_new import change_config_datasets
```

## 📚 API Reference

See individual module docstrings for complete API documentation:

- [DatasetRegistry](dataset_registry.py) - Dataset management
- [ConfigLoader](config_loader.py) - Configuration loading
- [HydraBridge](hydra_bridge.py) - Hydra compatibility layer

## ⚠️ Important Notes

1. **Backward Compatibility**: The original `utils_pipelines.py` is NOT modified. Use `utils_pipelines_new.py` for new code.

2. **Hydra Independence**: This module works alongside Hydra, not replacing it. Hydra is still used for training configs within champollion_V1.

3. **Dataset Discovery**: DatasetRegistry automatically discovers all `.yaml` files in `configs/dataset/` recursively.

4. **No Code Generation**: Unlike the old system, datasets are loaded dynamically - no code generation needed!

## 🤝 Integration with Main Pipeline

The main pipeline orchestrator (`main.py` at project root) can use this module to manage champollion_V1 configurations:

```python
# In main.py or pipeline stages
from champollion_pipeline.external.champollion_V1.contrastive.config_manager import ConfigLoader

loader = ConfigLoader()

# Get datasets for evaluation
datasets = loader.load_datasets_from_names(
    dataset_names=config.dataset.datasets,
    datasets_root=config.dataset.datasets_root
)
```

This creates a clean separation:
- **Main pipeline** (`main.py`): Uses simple YAML configs (no Hydra)
- **Champollion V1**: Uses Hydra internally, but configs are managed externally via config_manager
