# Lightweight Network for SFD by Attention-Aggregated Receptive-Field-Wise Feature

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Update the sensor paths in `config.yaml`:

```yaml
TRAIN_SENSOR_PATH:
TEST_SENSOR_PATH:
```

2. Run the training:

```bash
python binary_classification.py -c config.yaml
```

## Note

Use the foreground dataset to train and test.
