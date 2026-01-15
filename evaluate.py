import argparse
import yaml
import torch
import os

from model import get_model
from data_loaders import get_data_loaders, FinPADDataset, TransformedDataset
from torch.utils.data import DataLoader
from transforms import get_transforms
from metrics import find_optimal_threshold


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(model, checkpoint_path, device):
    """Load model state from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Checkpoint loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    return model


def get_test_loader(config):
    """Get test data loader based on config."""
    transform = get_transforms(config['TRANSFORM_TYPE'])
    test_dataset = FinPADDataset(config['TEST_SENSOR_PATH'], train=False, multiclass=False)
    test_set = TransformedDataset(test_dataset, transform['Test'])

    use_pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_set,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=use_pin_memory
    )
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of test batches: {len(test_loader)}")
    return test_loader


def evaluate(model, test_loader, device, based_on='ace'):
    """Evaluate model on test set."""
    model.eval()
    test_labels = []
    test_probabilities = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            normalized_outputs = (outputs + 1) / 2.0
            probabilities = normalized_outputs

            test_probabilities.append(probabilities.cpu())
            test_labels.append(labels.cpu())

    test_labels = torch.cat(test_labels).numpy()
    test_probabilities = torch.cat(test_probabilities).numpy()

    # Find optimal threshold and compute metrics
    threshold, apcer, bpcer, ace, accuracy = find_optimal_threshold(
        test_labels, test_probabilities, based_on=based_on
    )

    results = {
        'threshold': threshold,
        'apcer': apcer,
        'bpcer': bpcer,
        'ace': ace,
        'accuracy': accuracy,
        'based_on': based_on
    }

    return results, test_labels, test_probabilities


def print_results(results):
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    print(f"APCER (Attack Presentation Classification Error Rate): {results['apcer']:.2f}%")
    print(f"BPCER (Bonafide Presentation Classification Error Rate): {results['bpcer']:.2f}%")
    print(f"ACE (Average Classification Error): {results['ace']:.2f}%")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Optimal Threshold (based on {results['based_on']}): {results['threshold']:.6f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (overrides config)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path to save results (optional)')
    parser.add_argument('--based_on', type=str, default='ace',
                        choices=['ace', 'apcer', 'bpcer', 'accuracy'],
                        help='Metric to use for optimal threshold selection (default: ace)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine checkpoint path
    checkpoint_path = args.checkpoint or config.get('MODEL_SAVE_PATH')
    if checkpoint_path is None:
        raise ValueError("Checkpoint path must be specified either via --ckpt or in config")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Set device - prioritize GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Enable CUDA optimizations for consistent input sizes
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("Using CPU (CUDA not available)")

    # Load model
    model = get_model()
    model = load_checkpoint(model, checkpoint_path, device)

    # Get test data loader
    test_loader = get_test_loader(config)

    # Evaluate
    results, labels, probabilities = evaluate(model, test_loader, device, args.based_on)

    # Print results
    print_results(results)

    # Save results if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        torch.save({
            'results': results,
            'labels': labels,
            'probabilities': probabilities,
            'config': config,
            'checkpoint_path': checkpoint_path
        }, args.output)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
