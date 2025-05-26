import boto3
import json
import os
import time
import logging
from paddleocr import PaddleOCR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

directory_path = './sample4/'

# List all image files in the directory
image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    file_path = os.path.join(directory_path, image_file)
    
    # 对示例图像执行 OCR 推理 
    result = ocr.predict(
        input=file_path)
    
    # 可视化结果并保存 json 结果
    for res in result:
        res.json()
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")