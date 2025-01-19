import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from schedulefree import RAdamScheduleFree
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime
from sklearn.model_selection import train_test_split
import json
from src.models.resnet import FlexibleResNet
from src.data.dataset import CustomDataset


class ImageNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(
            'runs') / datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(self.log_dir)
        self._setup()

    def _split_dataset(self, annotation_file, test_size=0.2):
        """アノテーションを学習用とテスト用に分割"""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # 画像IDで分割
        image_ids = [img['id'] for img in annotations['images']]
        train_ids, test_ids = train_test_split(image_ids, test_size=test_size)

        # 訓練用とテスト用のアノテーションを作成
        train_annotations = {'images': [], 'annotations': [],
                             'categories': annotations['categories']}
        test_annotations = {'images': [], 'annotations': [],
                            'categories': annotations['categories']}

        for img in annotations['images']:
            if img['id'] in train_ids:
                train_annotations['images'].append(img)
            else:
                test_annotations['images'].append(img)

        for ann in annotations['annotations']:
            if ann['image_id'] in train_ids:
                train_annotations['annotations'].append(ann)
            else:
                test_annotations['annotations'].append(ann)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        train_path = Path(annotation_file).parent / \
            f'train_annotations_{timestamp}.json'
        test_path = Path(annotation_file).parent / \
            f'test_annotations_{timestamp}.json'

        for path, data in [(train_path, train_annotations), (test_path, test_annotations)]:
            with open(path, 'w') as f:
                json.dump(data, f)

        return train_path, test_path

    def _setup(self):
        """データとモデルの初期化"""
        train_path, test_path = self._split_dataset(
            self.args.annotation_file,
            test_size=getattr(self.args, 'test_size', 0.2)
        )

        # データセットの作成
        self.train_dataset = CustomDataset(
            root_dir=str(self.args.data_dir),
            annotation_file=str(train_path),
            max_size=256
        )

        self.test_dataset = CustomDataset(
            root_dir=str(self.args.data_dir),
            annotation_file=str(test_path),
            max_size=256
        )

        # DataLoaderの設定
        batch_size = getattr(self.args, 'batch_size', 32)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # モデルの設定
        self.model = FlexibleResNet(
            num_classes=self.train_dataset.num_classes).to(self.device)

        # クラスの重みを計算
        class_counts = self.train_dataset.get_class_counts()
        pos_weight = torch.FloatTensor(
            [len(self.train_dataset) / (count + 1e-5) for count in class_counts]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # オプティマイザーの設定
        self.optimizer = RAdamScheduleFree(
            self.model.parameters(),
            lr=getattr(self.args, 'lr', 0.001),
            weight_decay=getattr(self.args, 'weight_decay', 5e-4)
        )

        print(f"\nSetup complete:")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

    def train_epoch(self):
        """1エポックの訓練を実行"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = (output > 0.5).float()
            total += target.numel()
            correct += (predicted == target).sum().item()

            if batch_idx % 50 == 0:
                print(
                    f"Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.3f}, Acc={100.*correct/total:.2f}%")
                self.writer.add_scalar(
                    'Train/BatchLoss', loss.item(), self.current_epoch * len(self.train_loader) + batch_idx)

        return total_loss / len(self.train_loader), 100. * correct / total

    def test(self):
        """評価を実行"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                test_loss += loss.item()
                predicted = (output > 0.5).float()
                total += target.numel()
                correct += (predicted == target).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total

        print(
            f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return test_loss, accuracy

    def train(self):
        """モデルの訓練を実行"""
        best_acc = 0
        self.current_epoch = 0

        for epoch in range(getattr(self.args, 'epochs', 100)):
            self.current_epoch = epoch
            print(f'\nEpoch {epoch+1}')

            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test()

            # TensorBoardにメトリクスを記録
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Test/Loss', test_loss, epoch)
            self.writer.add_scalar('Test/Accuracy', test_acc, epoch)

            # モデルの保存
            if test_acc > best_acc:
                best_acc = test_acc
                save_dir = Path(getattr(self.args, 'save_dir', 'models'))
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc
                }, save_dir / 'best_model.pth')

        self.writer.close()
