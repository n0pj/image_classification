from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
import torch
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import lru_cache


class CustomDataset(Dataset):
    """マルチラベル画像分類のためのカスタムデータセット"""

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        max_size: int = 256
    ):
        self.root_dir = Path(root_dir)
        self.max_size = max_size

        # COCOフォーマットのアノテーションを読み込む
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # カテゴリーIDからインデックスへのマッピング
        self.categories = {cat['id']: i for i,
                           cat in enumerate(annotations['categories'])}
        self.classes = [cat['name'] for cat in sorted(
            annotations['categories'], key=lambda x: self.categories[x['id']])]
        self.num_classes = len(self.classes)

        # 画像とラベルのペアを作成
        self.samples = []
        image_annotations = {}

        # 画像ごとのアノテーションを集約
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(
                self.categories[ann['category_id']])

        # 画像情報とラベルを結合
        for img in annotations['images']:
            img_id = img['id']
            if img_id in image_annotations:
                img_path = self.root_dir / img['file_name']
                if img_path.exists():
                    labels = np.zeros(self.num_classes)
                    for cat_idx in image_annotations[img_id]:
                        labels[cat_idx] = 1
                    self.samples.append((str(img_path), labels))

        # デフォルトのデータ拡張設定
        self.transform = transform or A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        print(
            f"\nDataset loaded: {len(self.samples)} images, {self.num_classes} classes")

    @lru_cache(maxsize=1000)
    def _load_image(self, img_path: str) -> np.ndarray:
        """画像の読み込みと前処理（LRUキャッシュ付き）"""
        image = Image.open(img_path).convert('RGB')
        # アスペクト比を保持してリサイズ
        width, height = image.size
        scale = self.max_size / float(max(width, height))
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height),
                                 Image.Resampling.LANCZOS)
        return np.array(image)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        try:
            image = self._load_image(img_path)
            augmented = self.transform(image=image)
            return augmented['image'], torch.FloatTensor(labels)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.samples)

    def get_class_counts(self):
        """各クラスのサンプル数を返す"""
        class_counts = np.zeros(self.num_classes)
        for _, labels in self.samples:
            class_counts += labels
        return class_counts
