"""Data loading voor Flowers dataset."""
import torch
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from torchvision import transforms


IMG_SIZE = 128
BATCHSIZE = 32
NUM_CLASSES = 5


class AugmentPreprocessor:
    """Past transforms toe op een batch images."""
    
    def __init__(self, transform: transforms.Compose):
        self.transform = transform
    
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X = [self.transform(x) for x in X]
        return torch.stack(X), torch.stack(y)


def create_transforms(img_size: int = IMG_SIZE) -> dict:
    """Maak train en validation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {"train": train_transform, "valid": valid_transform}


def load_flowers_data(batchsize: int = BATCHSIZE, img_size: int = IMG_SIZE):
    """Laad Flowers dataset met augmentatie.
    
    Returns:
        tuple: (train_streamer, valid_streamer)
    """
    flowersfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    flowersfactory.settings.img_size = (img_size + 64, img_size + 64)
    
    streamers = flowersfactory.create_datastreamer(batchsize=batchsize)
    train = streamers["train"]
    valid = streamers["valid"]
    
    transforms_dict = create_transforms(img_size)
    train.preprocessor = AugmentPreprocessor(transforms_dict["train"])
    valid.preprocessor = AugmentPreprocessor(transforms_dict["valid"])
    
    logger.info(f"Train batches: {len(train)}, Valid batches: {len(valid)}")
    return train, valid


def get_device() -> torch.device:
    """Bepaal beste beschikbare device."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

