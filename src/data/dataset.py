from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import datetime
import json
import numpy as np
import torch


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
    def __init__(self, root_dir, annotation_file, transform=None, max_size=256):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_size = max_size

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

        self._print_dataset_info()

    def _print_dataset_info(self):
        """データセットの情報を表示"""
        print(f"\nDataset Info for {self.root_dir}:")
        print(f"Total images found: {len(self.samples)}")

        # クラスごとの画像数をカウント
        class_counts = {}
        for _, class_idx in self.samples:
            class_name = self.classes[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("\nImages per class:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = maintain_aspect_ratio_resize(image, self.max_size)

            if self.transform:
                image = self.transform(image)

            return image, torch.FloatTensor(labels)

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # エラーが発生した場合、データセット内の別の有効な画像を返す
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
