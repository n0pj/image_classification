import sqlite3
from datetime import datetime
import json
from pathlib import Path


class ImageDatabase:
    def __init__(self, db_path="images.db"):
        """データベースを初期化する

        Args:
            db_path (str): データベースファイルのパス
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """必要なテーブルを作成する"""
        cursor = self.conn.cursor()

        # 画像テーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # タグテーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        # 画像とタグの関連テーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_tags (
            image_id INTEGER,
            tag_id INTEGER,
            FOREIGN KEY (image_id) REFERENCES images (id),
            FOREIGN KEY (tag_id) REFERENCES tags (id),
            PRIMARY KEY (image_id, tag_id)
        )
        ''')

        self.conn.commit()

    def add_image(self, file_path: str, width: int, height: int) -> int:
        """画像をデータベースに追加する

        Args:
            file_path (str): 画像ファイルのパス
            width (int): 画像の幅
            height (int): 画像の高さ

        Returns:
            int: 追加された画像のID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO images (file_path, width, height) VALUES (?, ?, ?)',
            (str(file_path), width, height)
        )
        self.conn.commit()

        cursor.execute(
            'SELECT id FROM images WHERE file_path = ?', (str(file_path),))
        return cursor.fetchone()[0]

    def add_tag(self, tag_name: str) -> int:
        """タグをデータベースに追加する

        Args:
            tag_name (str): タグ名

        Returns:
            int: 追加されたタグのID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO tags (name) VALUES (?)', (tag_name,))
        self.conn.commit()

        cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
        return cursor.fetchone()[0]

    def link_image_tag(self, image_id: int, tag_id: int):
        """画像とタグを関連付ける

        Args:
            image_id (int): 画像ID
            tag_id (int): タグID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO image_tags (image_id, tag_id) VALUES (?, ?)',
            (image_id, tag_id)
        )
        self.conn.commit()

    def get_image_tags(self, image_id: int) -> list[str]:
        """画像に関連付けられたタグを取得する

        Args:
            image_id (int): 画像ID

        Returns:
            list[str]: タグのリスト
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT tags.name
            FROM tags
            JOIN image_tags ON tags.id = image_tags.tag_id
            WHERE image_tags.image_id = ?
        ''', (image_id,))
        return [row[0] for row in cursor.fetchall()]

    def export_coco(self, output_path: str):
        """データベースの内容を COCO 形式でエクスポートする

        Args:
            output_path (str): 出力 JSON ファイルのパス
        """
        cursor = self.conn.cursor()

        # 画像情報を取得
        cursor.execute('SELECT id, file_path, width, height FROM images')
        images = [
            {
                "id": row[0],
                "file_name": row[1],  # 絶対パスをそのまま使用
                "width": row[2],
                "height": row[3],
            }
            for row in cursor.fetchall()
        ]

        # タグ情報を取得
        cursor.execute('SELECT id, name FROM tags')
        categories = [
            {
                "id": row[0],
                "name": row[1],
                "supercategory": "tag"
            }
            for row in cursor.fetchall()
        ]

        # アノテーション情報を生成
        annotations = []
        for img in images:
            cursor.execute('''
                SELECT tags.id, tags.name
                FROM tags
                JOIN image_tags ON tags.id = image_tags.tag_id
                WHERE image_tags.image_id = ?
            ''', (img["id"],))

            for tag_id, tag_name in cursor.fetchall():
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": img["id"],
                    "category_id": tag_id,
                    "area": img["width"] * img["height"],
                    "bbox": [0, 0, img["width"], img["height"]],
                    "iscrowd": 0
                })

        coco_format = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "Exported from Image Annotation Database",
                "date_created": datetime.now().isoformat()
            },
            "images": images,
            "categories": categories,
            "annotations": annotations
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, ensure_ascii=False, indent=2)

    def get_statistics(self) -> dict:
        """データベースの統計情報を取得する

        Returns:
            dict: 統計情報を含む辞書
        """
        cursor = self.conn.cursor()

        # 総画像数
        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]

        # 総タグ数
        cursor.execute('SELECT COUNT(*) FROM tags')
        total_tags = cursor.fetchone()[0]

        # タグごとの画像数とランキング
        cursor.execute('''
            SELECT tags.name, COUNT(image_tags.image_id) as count
            FROM tags
            LEFT JOIN image_tags ON tags.id = image_tags.tag_id
            GROUP BY tags.id
            ORDER BY count DESC
        ''')
        tag_counts = [(row[0], row[1]) for row in cursor.fetchall()]

        return {
            "total_images": total_images,
            "total_tags": total_tags,
            "tag_counts": tag_counts
        }

    def __del__(self):
        """デストラクタ：データベース接続を閉じる"""
        if hasattr(self, 'conn'):
            self.conn.close()
