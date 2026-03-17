from torchvision import transforms


def get_train_transforms(img_size=224):
    resize_for_crop = int(img_size * 1.2)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.65, 1.0),
                ratio=(0.75, 1.3333),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.05,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.Resize((resize_for_crop, resize_for_crop)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )


def get_eval_transforms(img_size=224):
    resize_size = int(img_size * 1.14)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
