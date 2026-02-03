# scripts/batch_import.py
"""批量导入商品数据"""

import os
import sys
import json
import csv
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from tqdm import tqdm

from src.storage.milvus_client import MilvusClientWrapper
from src.encoders.image_encoder import AliyunImageEncoder
from src.encoders.text_encoder import ZhipuTextEncoder
from src.generators.description import ZhipuDescriptionGenerator


@dataclass
class ProductInput:
    """商品输入"""
    product_code: str
    category: str
    image_path: str
    description_human: Optional[str] = None


class BatchImporter:
    """批量导入器"""
    
    def __init__(
        self,
        milvus_client: MilvusClientWrapper,
        image_encoder: AliyunImageEncoder,
        text_encoder: ZhipuTextEncoder,
        desc_generator: ZhipuDescriptionGenerator,
        batch_size: int = 10
    ):
        self.milvus = milvus_client
        self.img_encoder = image_encoder
        self.text_encoder = text_encoder
        self.desc_gen = desc_generator
        self.batch_size = batch_size
    
    async def import_from_csv(self, csv_path: str):
        """从 CSV 文件导入"""
        products = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append(ProductInput(
                    product_code=row['product_code'],
                    category=row['category'],
                    image_path=row['image_path'],
                    description_human=row.get('description_human', '')
                ))
        await self._import_products(products)
    
    async def import_from_json(self, json_path: str):
        """从 JSON 文件导入"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = [
            ProductInput(
                product_code=p['product_code'],
                category=p['category'],
                image_path=p['image_path'],
                description_human=p.get('description_human', '')
            )
            for p in data['products']
        ]
        await self._import_products(products)
    
    async def import_from_directory(self, data_dir: str):
        """从目录结构导入"""
        products = []
        data_path = Path(data_dir)
        
        for category_dir in data_path.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for image_file in category_dir.glob(ext):
                    products.append(ProductInput(
                        product_code=image_file.stem,
                        category=category,
                        image_path=str(image_file),
                        description_human=''
                    ))
        
        await self._import_products(products)
    
    async def _import_products(self, products: List[ProductInput]):
        """批量处理并导入商品"""
        print(f"Total products to import: {len(products)}")
        
        for i in tqdm(range(0, len(products), self.batch_size), desc="Importing"):
            batch = products[i:i + self.batch_size]
            records = await self._process_batch(batch)
            
            if records:
                self.milvus.insert(records)
        
        print(f"Successfully imported {len(products)} products!")
    
    async def _process_batch(self, batch: List[ProductInput]) -> List[Dict]:
        """处理一批商品"""
        tasks = [self._process_single(p) for p in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        records = []
        for product, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"\nError processing {product.product_code}: {result}")
                continue
            records.append(result)
        
        return records
    
    async def _process_single(self, product: ProductInput) -> Dict:
        """处理单个商品"""
        # 读取图片
        with open(product.image_path, 'rb') as f:
            image_bytes = f.read()
        
        # 并行生成描述和图像 embedding
        desc_task = self.desc_gen.generate(image_bytes)
        img_emb_task = self.img_encoder.encode(image_bytes)
        
        description_ai, image_embedding = await asyncio.gather(desc_task, img_emb_task)
        
        # 生成文本 embedding
        text_embedding = await self.text_encoder.encode(description_ai)
        
        return {
            "product_code": product.product_code,
            "category": product.category,
            "description_human": product.description_human or "",
            "description_ai": description_ai,
            "image_embedding": image_embedding.tolist(),
            "text_embedding": text_embedding.tolist(),
            "image_url": product.image_path,
            "created_at": int(time.time())
        }


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量导入商品数据')
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录路径')
    parser.add_argument('--format', '-f', choices=['csv', 'json', 'dir'], default='csv',
                        help='输入格式: csv, json, dir')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='批处理大小')
    args = parser.parse_args()
    
    load_dotenv()
    
    # 初始化组件
    milvus_client = MilvusClientWrapper(
        uri=os.getenv("ZILLIZ_CLOUD_URI"),
        token=os.getenv("ZILLIZ_CLOUD_TOKEN"),
        collection_name=os.getenv("ZILLIZ_CLOUD_COLLECTION", "products")
    )
    image_encoder = AliyunImageEncoder(os.getenv("ALIYUN_DASHSCOPE_API_KEY"))
    text_encoder = ZhipuTextEncoder(os.getenv("ZHIPU_API_KEY"))
    desc_generator = ZhipuDescriptionGenerator(os.getenv("ZHIPU_API_KEY"))
    
    importer = BatchImporter(
        milvus_client=milvus_client,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        desc_generator=desc_generator,
        batch_size=args.batch_size
    )
    
    try:
        if args.format == 'csv':
            await importer.import_from_csv(args.input)
        elif args.format == 'json':
            await importer.import_from_json(args.input)
        else:
            await importer.import_from_directory(args.input)
    finally:
        milvus_client.close()
        await image_encoder.close()
        await text_encoder.close()
        await desc_generator.close()


if __name__ == "__main__":
    asyncio.run(main())
