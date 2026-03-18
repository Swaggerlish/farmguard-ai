from PIL import Image
from torchvision import transforms


def get_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def preprocess_image(file_obj, img_size: int = 224):
    image = Image.open(file_obj).convert("RGB")
    transform = get_eval_transform(img_size)
    tensor = transform(image).unsqueeze(0)
    return image, tensor