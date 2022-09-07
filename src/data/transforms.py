import torchvision.transforms as transforms

TRAIN_MEAN = [0.3683037974792629, 0.2545500826256079, 0.21100532802110092]
TRAIN_STD = [0.32256124182067264, 0.23441272870730917, 0.21096266280494616]


def get_transforms():
    normalize = transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)

    train_transform = transforms.Compose(
        [
            # TODO: Elif is this the right image size?
            # transforms.RandomResizedCrop((540, 960), scale=(0.8, 1.2), ratio=(0.75, 1.33)),
            transforms.RandomResizedCrop(
            #    (540 // 2, 960 // 2), scale=(0.8, 1.2), ratio=(0.75, 1.33)
                (480, 625), scale=(0.8, 1.2), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((180)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose([
        transforms.RandomResizedCrop((480, 625)),
        transforms.ToTensor(), normalize])
    return train_transform, val_transform
