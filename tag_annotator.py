from pathlib import Path
from PIL import Image
from database import ImageDatabase
import argparse
from tqdm import tqdm


def extract_tags_from_path(path: Path, base_dir: Path) -> list[str]:
    """パスからタグを抽出する

    Args:
        path (Path): ディレクトリパス
        base_dir (Path): 基準となるディレクトリパス

    Returns:
        list[str]: タグのリスト
    """
    tags = []

    try:
        # base_dirからの相対パスを取得
        rel_path = path.resolve().relative_to(base_dir.resolve())

        # 直接の親ディレクトリからタグを抽出
        if str(rel_path) != '.':
            for tag in rel_path.name.split():
                tags.append(tag)
    except ValueError:
        # base_dirの外側のパスは無視
        pass

    return list(set(tags))  # 重複を削除


def process_directory(base_dir: Path, db: ImageDatabase):
    """ディレクトリを再帰的に処理する

    Args:
        base_dir (Path): 基準ディレクトリ
        db (ImageDatabase): データベースインスタンス
    """
    # 画像ファイルの拡張子
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    # 全画像ファイルを収集
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(base_dir.rglob(f"*{ext}"))

    print(f"合計 {len(image_files)} 個の画像ファイルを処理します")

    # 各画像ファイルを処理
    for img_path in tqdm(image_files, desc="画像処理中"):
        try:
            # 画像サイズを取得し、有効な画像かを確認
            with Image.open(img_path) as img:
                # 画像フォーマットを確認
                if img.format not in ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']:
                    print(f"警告: {img_path} は未対応の画像フォーマットです: {img.format}")
                    continue

                # 画像サイズを取得して有効性を確認
                try:
                    width, height = img.size
                    # 画像をデータベースに追加（絶対パスに変換）
                    image_id = db.add_image(
                        str(img_path.resolve()), width, height)
                except Exception as e:
                    print(f"警告: {img_path} は破損しているか無効な画像ファイルです: {e}")
                    continue

            # パスからタグを抽出して関連付け
            tags = extract_tags_from_path(img_path.parent, base_dir)
            for tag in tags:
                tag_id = db.add_tag(tag)
                db.link_image_tag(image_id, tag_id)

        except Exception as e:
            print(f"警告: {img_path} の処理中にエラーが発生しました: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="画像ファイルにタグを付与し、COCO 形式でエクスポートする")
    parser.add_argument("input_dir", type=str, help="画像ファイルを含むディレクトリ")
    parser.add_argument("--output", "-o", type=str,
                        default="annotations.json", help="出力 JSON ファイルのパス")
    parser.add_argument("--db", type=str, default="images.db",
                        help="SQLite データベースファイルのパス")
    parser.add_argument("--reset", action="store_true",
                        help="データベースを初期化する")
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    if not base_dir.exists():
        print(f"エラー: ディレクトリ {base_dir} が存在しません")
        return

    print(f"データベース {args.db} を初期化中...")
    if args.reset and Path(args.db).exists():
        print("既存のデータベースを削除中...")
        Path(args.db).unlink()
    db = ImageDatabase(args.db)

    print(f"ディレクトリ {base_dir} を処理中...")
    process_directory(base_dir, db)

    print(f"アノテーションを {args.output} にエクスポート中...")
    db.export_coco(args.output)

    # 統計情報の表示
    stats = db.get_statistics()
    print("\n=== 統計情報 ===")
    print(f"総画像数: {stats['total_images']}")
    print(f"総タグ数: {stats['total_tags']}")
    print("\n=== タグ使用頻度ランキング ===")
    for tag_name, count in stats['tag_counts']:
        print(f"- {tag_name}: {count}枚")

    print("\n完了しました！")


if __name__ == "__main__":
    main()
