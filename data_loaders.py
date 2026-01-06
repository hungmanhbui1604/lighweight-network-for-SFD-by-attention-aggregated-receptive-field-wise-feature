import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transforms import get_transforms

class FinPADDataset(Dataset):
    def __init__(self, sensor_path, train=True, multiclass=False):
        self.samples = []
        self.label_map = {}
        self.supported_extensions = ('.png', '.bmp')
        
        if multiclass:
            self.label_map = {'Live': 0}
        else:
            self.label_map = {'Live': 0, 'Spoof': 1}

        phase = 'Train' if train else 'Test'
        phase_path = os.path.join(sensor_path, phase)
        if multiclass:
            self._load_multiclass(phase_path)
        else:
            self._load_binary(phase_path)

    def _load_binary(self, phase_path):
        for label_name in ['Live', 'Spoof']:
            label_id = self.label_map[label_name]
            label_path = os.path.join(phase_path, label_name)
            if label_name == 'Live':
                for img_file in tqdm(os.listdir(label_path), desc=f"Loading Live"):
                    if img_file.endswith(self.supported_extensions):
                        image_path = os.path.join(label_path, img_file)
                        self.samples.append((image_path, label_id))
            elif label_name == 'Spoof':
                for material in os.listdir(label_path):
                    material_path = os.path.join(label_path, material)
                    for img_file in tqdm(os.listdir(material_path), desc=f"Loading {material}"):
                        if img_file.endswith(self.supported_extensions):
                            image_path = os.path.join(material_path, img_file)
                            self.samples.append((image_path, label_id))

    def _load_multiclass(self, phase_path):
        live_path = os.path.join(phase_path, 'Live')
        for img_file in tqdm(os.listdir(live_path), desc=f"Loading Live"):
            if img_file.endswith(self.supported_extensions):
                image_path = os.path.join(live_path, img_file)
                self.samples.append((image_path, self.label_map['Live']))

        spoof_path = os.path.join(phase_path, 'Spoof')
        materials = sorted([material for material in os.listdir(spoof_path)])
        
        for i, material in enumerate(materials, 1):
            self.label_map[material] = i
            material_path = os.path.join(spoof_path, material)
            for img_file in tqdm(os.listdir(material_path), desc=f"Loading {material}"):
                if img_file.endswith(('.png', '.bmp')):
                    image_path = os.path.join(material_path, img_file)
                    self.samples.append((image_path, self.label_map[material]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('L')
        return image, label

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def split_dataset(dataset: Dataset, val_split: float = 0.2, seed: int = 42):    
    generator = torch.Generator().manual_seed(seed)

    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    return random_split(dataset, [train_size, val_size], generator=generator)


def get_data_loaders(
    train_sensor_path: str,
    test_sensor_path: str,
    multiclass: bool,
    transform: dict,
    batch_size: int,
    num_workers: int = 0,
    val_split: float = 0.2,
    seed: int = 42,
):
    # create dataset
    if train_sensor_path == test_sensor_path: # intra-sensor
        sensor_path = train_sensor_path
        print(f"Creating intra-sensor {sensor_path} dataset")
        print("Loading train data")
        train_dataset = FinPADDataset(sensor_path, train=True, multiclass=multiclass)
        print("Loading test data")
        test_dataset = FinPADDataset(sensor_path, train=False, multiclass=multiclass)
    else: # cross-sensor
        print(f"Creating cross-sensor {train_sensor_path}-{test_sensor_path} dataset")
        print("Loading train data")
        train_dataset = FinPADDataset(train_sensor_path, train=True, multiclass=multiclass)
        print("Loading test data")
        test_dataset = FinPADDataset(test_sensor_path, train=False, multiclass=multiclass)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    
    # get label map
    label_map = train_dataset.label_map

    use_pin_memory = True if torch.cuda.is_available() else False
    # Train phase
    train_subset, val_subset = split_dataset(train_dataset, val_split=val_split, seed=seed)
    train_set = TransformedDataset(train_subset, transform['Train'])
    val_set = TransformedDataset(val_subset, transform['Test'])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of val batches: {len(val_loader)}")

    # Test phase
    test_set = TransformedDataset(test_dataset, transform['Test'])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    print(f"Number of test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader, label_map