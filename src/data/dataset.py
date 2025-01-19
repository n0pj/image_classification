from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import datetime
import json
import numpy as np
import torch
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Union, TypeVar
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from functools import lru_cache


class COCOSplitter:
    def __init__(self, annotation_file, test_size=0.2, random_state=42):
        self.annotation_file = Path(annotation_file)
        self.test_size = test_size
        self.random_state = random_state

        # アノテーションファイルの存在確認
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}")

        # アノテーションの読み込み
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def split_dataset(self):
        """アノテーションを学習用とテスト用に分割"""
        print("Splitting annotations into train and test sets...")

        # 画像IDのリストを作成
        image_ids = [img['id'] for img in self.annotations['images']]

        # 画像IDを分割
        train_ids, test_ids = train_test_split(
            image_ids,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # 訓練用と評価用のアノテーションを作成
        train_annotations = {
            'images': [],
            'annotations': [],
            'categories': self.annotations['categories']
        }

        test_annotations = {
            'images': [],
            'annotations': [],
            'categories': self.annotations['categories']
        }

        # 画像情報を分割
        image_id_to_split = {
            id: 'train' if id in train_ids else 'test' for id in image_ids}

        for img in self.annotations['images']:
            if image_id_to_split[img['id']] == 'train':
                train_annotations['images'].append(img)
            else:
                test_annotations['images'].append(img)

        # アノテーションを分割
        for ann in self.annotations['annotations']:
            if image_id_to_split[ann['image_id']] == 'train':
                train_annotations['annotations'].append(ann)
            else:
                test_annotations['annotations'].append(ann)

        print(f"Total images: {len(image_ids)}")
        print(f"Training images: {len(train_annotations['images'])}")
        print(f"Test images: {len(test_annotations['images'])}")

        # 分割したアノテーションを保存
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        train_path = self.annotation_file.parent / \
            f'train_annotations_{timestamp}.json'
        test_path = self.annotation_file.parent / \
            f'test_annotations_{timestamp}.json'

        with open(train_path, 'w') as f:
            json.dump(train_annotations, f, indent=2)

        with open(test_path, 'w') as f:
            json.dump(test_annotations, f, indent=2)

        print(f"\nSplit complete!")
        print(f"Train annotations: {train_path}")
        print(f"Test annotations: {test_path}")

        return train_path, test_path


class CustomDataset(Dataset):
    """マルチラベル画像分類のためのカスタムデータセット

    特徴:
    - COCOフォーマットのアノテーションに対応
    - キャッシュ機構による高速なデータロード
    - メモリ効率の良い画像処理
    - 豊富なデータ拡張
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        max_size: int = 256,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            root_dir: 画像ファイルが格納されているディレクトリ
            annotation_file: COCOフォーマットのアノテーションファイルパス
            transform: カスタムのデータ拡張
            max_size: 画像の最大サイズ
            use_cache: キャッシュを使用するかどうか
            cache_dir: キャッシュディレクトリのパス（Noneの場合はroot_dir/.cache）
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_size = max_size
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path(
            root_dir) / '.cache'

        # キャッシュディレクトリの設定
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache: Dict[str, torch.Tensor] = {}

        # COCOフォーマットのアノテーションを読み込む
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # カテゴリーIDからインデックスへのマッピング
        self.categories = {cat['id']: i for i,
                           cat in enumerate(self.annotations['categories'])}
        self.classes = [cat['name'] for cat in sorted(
            self.annotations['categories'], key=lambda x: self.categories[x['id']])]
        self.num_classes = len(self.classes)

        # 画像とラベルのペアを作成
        self.samples = []
        image_annotations = {}

        # 画像ごとのアノテーションを集約
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(
                self.categories[ann['category_id']])

        # 画像情報とラベルを結合
        for img in self.annotations['images']:
            img_id = img['id']
            if img_id in image_annotations:
                img_path = self.root_dir / img['file_name']
                if img_path.exists():
                    # マルチラベルのone-hotエンコーディング
                    labels = np.zeros(self.num_classes)
                    for cat_idx in image_annotations[img_id]:
                        labels[cat_idx] = 1
                    self.samples.append((str(img_path), labels))

        # 基本的なデータ拡張の設定
        self.base_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.ElasticTransform(p=1),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # キャッシュの初期化
        if self.use_cache:
            self._initialize_cache()

        # データセット情報の表示
        self._print_dataset_info()

    def _print_dataset_info(self):
        """データセットの情報を表示"""
        print(f"\nDataset Info for {self.root_dir}:")
        print(f"Total images found: {len(self.samples)}")

        # クラスごとの画像数をカウント
        class_counts = np.zeros(self.num_classes)
        for _, labels in self.samples:
            class_counts += labels

        print("\nImages per class:")
        for i, count in enumerate(class_counts):
            print(f"  {self.classes[i]}: {int(count)}")
        print()

    def get_class_counts(self):
        """各クラスのサンプル数を返す"""
        class_counts = np.zeros(self.num_classes)
        for _, labels in self.samples:
            class_counts += labels
        return class_counts

    def __len__(self):
        return len(self.samples)

    def _initialize_cache(self):
        """キャッシュの初期化とチェック"""
        if not self.use_cache:
            return

        self.cache: Dict[str, torch.Tensor] = {}
        for img_path, _ in self.samples:
            cache_path = self._get_cache_path(img_path)
            if cache_path.exists():
                try:
                    with cache_path.open('rb') as f:
                        self.cache[img_path] = pickle.load(f)
                except Exception as e:
                    logging.warning(
                        f"Failed to load cache for {img_path}: {e}")

    def _get_cache_path(self, img_path: str) -> Path:
        """画像パスからキャッシュファイルのパスを生成"""
        hash_name = hashlib.md5(img_path.encode()).hexdigest()
        return self.cache_dir / f"{hash_name}.pkl"

    @lru_cache(maxsize=1000)
    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """画像の読み込みと前処理（LRUキャッシュ付き）"""
        image = Image.open(img_path).convert('RGB')
        image = maintain_aspect_ratio_resize(image, self.max_size)
        return np.array(image)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, labels = self.samples[idx]

        try:
            # キャッシュからの読み込みを試行
            if self.use_cache and img_path in self.cache:
                image = self.cache[img_path]
            else:
                # 画像の読み込みと前処理
                image = self._load_and_preprocess_image(img_path)

                # データ拡張の適用
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                else:
                    augmented = self.base_transform(image=image)
                    image = augmented['image']

                # キャッシュの保存
                if self.use_cache:
                    cache_path = self._get_cache_path(img_path)
                    with cache_path.open('wb') as f:
                        pickle.dump(image, f)
                    self.cache[img_path] = image

            return image, torch.FloatTensor(labels)

        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))


def maintain_aspect_ratio_resize(image, max_size=256):
    """アスペクト比を保持しながら、指定サイズ以下にリサイズする"""
    width, height = image.size

    # 大きい方の辺を基準にスケールを計算
    scale = max_size / float(max(width, height))

    # 元のサイズが既に max_size 以下の場合はリサイズ不要
    if scale >= 1:
        return image

    # 新しいサイズを計算 ( アスペクト比を保持 )
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
