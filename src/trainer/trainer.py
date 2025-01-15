import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from schedulefree import RAdamScheduleFree
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path
from src.data.dataset import COCOSplitter
from src.models.resnet import FlexibleResNet
from src.data.dataset import CustomDataset


class ImageNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # アノテーションの分割
        splitter = COCOSplitter(
            annotation_file=args.annotation_file,
            test_size=args.test_size if hasattr(args, 'test_size') else 0.2,
            random_state=args.random_state if hasattr(
                args, 'random_state') else 42
        )

        # 分割したアノテーションのパスを保存
        self.train_annotation, self.test_annotation = splitter.split_dataset()

        # TensorBoard 用の writer 初期化
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = Path('runs') / f'imagenet_{current_time}'
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")

        # データとモデルのセットアップ
        self._setup_data()
        self._setup_model()

    def _setup_data(self):
        # データ拡張とノーマライゼーションの定義
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # train_dir と test_dir が存在することを確認
        # アノテーションファイルの存在を確認
        if not Path(self.train_annotation).exists():
            raise FileNotFoundError(
                f"Training annotation file not found: {self.train_annotation}")
        if not Path(self.test_annotation).exists():
            raise FileNotFoundError(
                f"Test annotation file not found: {self.test_annotation}")

        # カスタムデータセットの読み込み
        self.train_dataset = CustomDataset(
            root_dir=str(self.args.data_dir),
            annotation_file=str(self.train_annotation),
            transform=train_transform,
            max_size=256
        )

        self.test_dataset = CustomDataset(
            root_dir=str(self.args.data_dir),
            annotation_file=str(self.test_annotation),
            transform=test_transform,
            max_size=256
        )

        # DataLoader の設定
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size if hasattr(
                self.args, 'batch_size') else 32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size if hasattr(
                self.args, 'batch_size') else 32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        self.num_classes = len(self.train_dataset.classes)
        print(f"\nDataset setup complete:")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.train_dataset.classes}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}\n")

    def _collate_fn(self, batch):
        """可変サイズの画像をバッチ化するための関数 ( 256x256 以下 )"""
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # 画像の高さと幅を取得
        heights = [img.shape[1] for img in images]
        widths = [img.shape[2] for img in images]

        # バッチ内の最大サイズを計算
        max_height = max(heights)
        max_width = max(widths)

        # パディングを追加して同じサイズにする
        padded_images = []
        for img in images:
            pad_height = max_height - img.shape[1]
            pad_width = max_width - img.shape[2]

            # パディングを追加 ( 右下に追加 )
            padded_img = F.pad(img, (0, pad_width, 0, pad_height),
                               mode='constant', value=0)
            padded_images.append(padded_img)

        # バッチとしてスタック
        images = torch.stack(padded_images)
        labels = torch.tensor(labels)

        return images, labels

    def _setup_model(self):
        # ResNet18 をベースにモデルを作成
        self.model = FlexibleResNet(num_classes=self.num_classes)

        # モデルを GPU に移動
        self.model = self.model.to(self.device)

        # オプティマイザーの設定
        self.optimizer = RAdamScheduleFree(
            self.model.parameters(),
            lr=self.args.lr if hasattr(self.args, 'lr') else 0.001,
            weight_decay=self.args.weight_decay if hasattr(
                self.args, 'weight_decay') else 5e-4
        )

        self.criterion = nn.BCEWithLogitsLoss()

        # TensorBoard のグラフ追加 ( オプション )
        try:
            # ダミー入力を同じデバイスに配置
            dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
            self.writer.add_graph(self.model, dummy_input)
        except Exception as e:
            print("Warning: Failed to add model graph to TensorBoard:", str(e))
            print("This is not critical and training will continue.")

    def train_epoch(self):
        self.model.train()
        self.optimizer.train()

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
            # マルチラベル分類の評価
            predicted = (output > 0.5).float()
            total += target.size(0) * target.size(1)  # 全ラベルの数
            correct += (predicted == target).sum().item()

            # バッチごとの損失を TensorBoard に記録
            if batch_idx % 100 == 0:
                step = len(self.train_loader) * self.current_epoch + batch_idx
                self.writer.add_scalar('Batch/Train Loss', loss.item(), step)
                self.writer.add_scalar('Batch/Train Accuracy',
                                       100. * correct / total, step)

                print(f'Train Batch: {batch_idx}/{len(self.train_loader)} '
                      f'Loss: {loss.item():.3f} '
                      f'Acc: {100.*correct/total:.2f}%')

        return total_loss / len(self.train_loader), 100. * correct / total

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        # クラスごとの正解数と総数を記録
        class_correct = torch.zeros(self.num_classes)
        class_total = torch.zeros(self.num_classes)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                test_loss += loss.item()
            # マルチラベル分類の評価
            predicted = (output > 0.5).float()
            total += target.size(0) * target.size(1)
            correct += (predicted == target).sum().item()

            # クラスごとの正解数を計算
            for i in range(self.num_classes):
                class_total[i] += target[:, i].sum().item()
                class_correct[i] += ((predicted[:, i] == 1)
                                     & (target[:, i] == 1)).sum().item()

                if batch_idx % 10 == 0:
                    print(f'Test Batch: {batch_idx}/{len(self.test_loader)} '
                          f'Loss: {loss.item():.3f} '
                          f'Acc: {100.*correct/total:.2f}%')

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total

        # クラスごとの精度を計算 ( データが存在するクラスのみ )
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, total, test_acc))

        print('Class-wise accuracy:')
        for i in range(self.num_classes):
            if class_total[i] > 0:  # データが存在するクラスのみ表示
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'Class {self.test_dataset.classes[i]}: {class_acc:.2f}% '
                      f'({int(class_correct[i])}/{int(class_total[i])})')
            else:
                print(f'Class {self.test_dataset.classes[i]}: No test samples')

            # TensorBoard に記録 ( データが存在しないクラスは0%として記録 )
            class_acc = 100 * class_correct[i] / max(class_total[i], 1)
            self.writer.add_scalar(f'Test/Class_{self.test_dataset.classes[i]}_Accuracy',
                                   class_acc,
                                   self.current_epoch)

        # 全体の精度を TensorBoard に記録
        self.writer.add_scalar('Test/Average_Loss',
                               test_loss, self.current_epoch)
        self.writer.add_scalar('Test/Average_Accuracy',
                               test_acc, self.current_epoch)

        return test_loss, test_acc

    def train(self):
        best_acc = 0
        self.current_epoch = 0

        print("\nStarting training...")
        print(f"Training on device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Classes: {self.train_dataset.classes}\n")

        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            print(f'\nEpoch: {epoch+1}/{self.args.epochs}')

            # エポックごとのデータ数を確認
            train_samples = sum([len(self.train_loader.dataset.samples[i:i+self.args.batch_size])
                                 for i in range(0, len(self.train_loader.dataset.samples),
                                                self.args.batch_size)])
            test_samples = sum([len(self.test_loader.dataset.samples[i:i+self.args.batch_size])
                                for i in range(0, len(self.test_loader.dataset.samples),
                                               self.args.batch_size)])

            print(f"Training samples this epoch: {train_samples}")
            print(f"Test samples this epoch: {test_samples}")

            if train_samples == 0:
                print("No training samples available. Skipping epoch.")
                continue

            train_loss, train_acc = self.train_epoch()

            if test_samples == 0:
                print("No test samples available. Skipping evaluation.")
                test_loss, test_acc = float('inf'), 0
            else:
                test_loss, test_acc = self.test()

            print(
                f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')

            # モデルの保存 ( テストデータがある場合のみ )
            if test_samples > 0 and test_acc > best_acc:
                print('Saving model...')
                best_acc = test_acc
                save_dir = Path(self.args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                    'classes': self.train_dataset.classes
                }, save_dir / 'best_model.pth')

        self.writer.close()
