import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_weak_augmentation(img_size=256):
    return A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3)])

def get_strong_augmentation(img_size=256):
    return A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1), A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1), A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1)], p=0.8), A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=1), A.GaussNoise(var_limit=(10, 50), p=1), A.MedianBlur(blur_limit=5, p=1)], p=0.3), A.CoarseDropout(max_holes=8, max_height=img_size // 8, max_width=img_size // 8, min_holes=1, min_height=img_size // 16, min_width=img_size // 16, fill_value=0, p=0.3), A.ElasticTransform(alpha=50, sigma=50 * 0.05, p=0.2), A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2)])

def get_validation_augmentation(img_size=256):
    return A.Compose([A.Resize(img_size, img_size)])

class DualAugmentation:

    def __init__(self, img_size=256):
        self.img_size = img_size
        self.geo_transform = A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3)])
        self.strong_extra = A.Compose([A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1), A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1), A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1)], p=0.8), A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=1), A.GaussNoise(var_limit=(10, 50), p=1)], p=0.3), A.CoarseDropout(max_holes=8, max_height=img_size // 8, max_width=img_size // 8, min_holes=1, fill_value=0, p=0.3)])

    def __call__(self, image, mask):
        geo_result = self.geo_transform(image=image, mask=mask)
        image_geo = geo_result['image']
        mask_geo = geo_result['mask']
        image_weak = image_geo.copy()
        strong_result = self.strong_extra(image=image_geo)
        image_strong = strong_result['image']
        return {'image_weak': image_weak, 'image_strong': image_strong, 'mask': mask_geo}
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    dual_aug = DualAugmentation(img_size=256)
    result = dual_aug(image, mask)
    print(f"Weak image shape: {result['image_weak'].shape}")
    print(f"Strong image shape: {result['image_strong'].shape}")
    print(f"Mask shape: {result['mask'].shape}")
