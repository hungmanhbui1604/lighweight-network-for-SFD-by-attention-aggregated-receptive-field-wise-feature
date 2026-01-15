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

3. Evaluate a trained model on the test set:

```bash
python evaluate.py -c config.yaml -ckpt ./checkpoints/model1.pth -o ./results/eval_results.pth --based_on ace
```

## Note

Use the foreground dataset to train and test.
