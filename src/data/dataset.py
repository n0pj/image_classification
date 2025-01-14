from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import datetime


class DatasetSplitter:
    def __init__(self, source_dir, test_size=0.2, random_state=42):
        self.source_dir = Path(source_dir)
        self.test_size = test_size
        self.random_state = random_state

        # 元のデータディレクトリの存在確認
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found: {self.source_dir}")

        # 分割データの保存先を元のディレクトリと同じ階層に作成
        self.output_base_dir = self.source_dir.parent / 'dataset_split'
        self.temp_train_dir = self.output_base_dir / 'train'
        self.temp_test_dir = self.output_base_dir / 'test'

    def split_dataset(self):
        """データセットを学習用とテスト用に分割"""
        print("Splitting dataset into train and test sets...")

        # 新しい分割用ディレクトリを作成 ( 既存のものは残す )
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_base_dir = self.source_dir.parent / \
            f'dataset_split_{timestamp}'
        self.temp_train_dir = self.output_base_dir / 'train'
        self.temp_test_dir = self.output_base_dir / 'test'

        # クラスごとにデータを分割
        class_paths = [d for d in self.source_dir.iterdir() if d.is_dir()]

        if not class_paths:
            raise ValueError(
                f"No class directories found in {self.source_dir}")

        for class_path in class_paths:
            print(f"\nProcessing class: {class_path.name}")

            # クラス内の全画像ファイルを収集 ( サブディレクトリを含む )
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(class_path.rglob(ext)))
                image_files.extend(list(class_path.rglob(ext.upper())))

            if not image_files:
                print(f"Warning: No images found in class {class_path.name}")
                continue

            # データ分割
            train_files, test_files = train_test_split(
                image_files,
                test_size=self.test_size,
                random_state=self.random_state
            )

            print(f"Total images: {len(image_files)}")
            print(f"Training images: {len(train_files)}")
            print(f"Test images: {len(test_files)}")

            # ファイルのコピー
            self._copy_files(train_files, class_path.name, is_train=True)
            self._copy_files(test_files, class_path.name, is_train=False)

        print(f"\nDataset split complete!")
        print(f"Train data: {self.temp_train_dir}")
        print(f"Test data: {self.temp_test_dir}")

        return self.temp_train_dir, self.temp_test_dir

    def _copy_files(self, files, class_name, is_train=True):
        """ファイルを対応するディレクトリにコピー"""
        target_dir = (
            self.temp_train_dir if is_train else self.temp_test_dir) / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for src_path in files:
            try:
                # サブディレクトリ構造を維持
                relative_path = src_path.relative_to(
                    self.source_dir / class_name)
                dst_path = target_dir / relative_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Warning: Failed to copy {src_path}: {e}")


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_size=256):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_size = max_size

        # トップレベルのディレクトリのみをクラスとして扱う
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.classes)}

        self.samples = []
        # 各クラスディレクトリを処理
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                class_idx = self.class_to_idx[class_dir.name]
                # サブディレクトリを含む全ての画像ファイルを再帰的に検索
                for img_path in class_dir.rglob('*'):
                    # 画像ファイルの拡張子をチェック
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((str(img_path), class_idx))

        # 見つかった画像の総数とクラスごとの内訳を表示
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
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = maintain_aspect_ratio_resize(image, self.max_size)

            if self.transform:
                image = self.transform(image)

            return image, label

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
