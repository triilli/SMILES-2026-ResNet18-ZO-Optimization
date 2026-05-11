import torchvision.transforms as T

# Per-channel mean and std computed on the CIFAR100 training set.
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool = True) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize(224),
            T.RandomHorizontalFlip(),
            # Add more augmentations here ↓
            T.RandomCrop(224, padding=28),
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])
    else:
        # Fixed validation pipeline — do not modify.
        return T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])