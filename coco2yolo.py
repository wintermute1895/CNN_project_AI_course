import os
import json
from tqdm import tqdm

def coco2yolo(coco_json_path, output_dir, image_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载COCO标注文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建类别ID映射（COCO ID → YOLO连续ID，从0开始）
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # 按图像ID分组标注
    image_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_anns:
            image_anns[image_id] = []
        image_anns[image_id].append(ann)

    # 处理每张图像
    for img in tqdm(coco_data['images'], desc=f"Processing {os.path.basename(image_dir)}"):
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        file_name = img['file_name']

        # 检查图像是否存在（可选）
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            print(f"警告：图像文件 {file_name} 不存在，跳过处理")
            continue

        # 获取当前图像的所有标注
        anns = image_anns.get(img_id, [])

        # 生成YOLO标签文件路径
        txt_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')

        with open(txt_path, 'w') as f_txt:
            for ann in anns:
                # 获取类别ID
                cat_id = categories[ann['category_id']]

                # 提取COCO bbox格式: [x_min, y_min, width, height]
                x, y, w, h = ann['bbox']

                # 转换为YOLO格式: [x_center, y_center, width, height]（归一化）
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # 写入文件
                f_txt.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

if __name__ == '__main__':
    # 配置路径（根据实际路径修改）
    datasets = [
        {
            "coco_json": "annotations/instances_train2014.json",
            "output_dir": "labels/train2014",
            "image_dir": "images/train2014"
        },
        {
            "coco_json": "annotations/instances_val2014.json",
            "output_dir": "labels/val2014",
            "image_dir": "images/val2014"
        }
    ]

    # 批量处理所有数据集
    for dataset in datasets:
        print(f"\n正在处理数据集: {dataset['image_dir']}")
        coco2yolo(
            coco_json_path=dataset['coco_json'],
            output_dir=dataset['output_dir'],
            image_dir=dataset['image_dir']
        )