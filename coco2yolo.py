import os
import json
from tqdm import tqdm
from pathlib import Path

def coco2yolo(coco_json_path, output_dir, image_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建类别ID映射和名称列表（按YOLO ID顺序）
    categories = {}
    class_names = []
    for idx, cat in enumerate(coco_data['categories']):
        categories[cat['id']] = idx
        class_names.append(cat['name'])  # 新增：记录类别名称
    
    # 新增：生成classes.txt文件（按YOLO ID顺序）
    classes_path = os.path.join(output_dir, "classes.txt")
    if not os.path.exists(classes_path):  # 避免重复写入
        with open(classes_path, 'w') as f_cls:
            f_cls.write('\n'.join(class_names))

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
                # 获取类别ID（新增错误处理）
                try:
                    cat_id = categories[ann['category_id']]
                except KeyError:
                    print(f"警告：发现未定义的类别ID {ann['category_id']}，已跳过")
                    continue

                # 转换坐标（保留边界检查）
                x, y, w, h = ann['bbox']
                x_center = max(0.0, min((x + w / 2) / img_w, 1.0))
                y_center = max(0.0, min((y + h / 2) / img_h, 1.0))
                w_norm = max(0.0, min(w / img_w, 1.0))
                h_norm = max(0.0, min(h / img_h, 1.0))

                # 写入文件（去除格式校验，直接写入）
                f_txt.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.resolve()
    
    datasets = [
        {
            "coco_json": str(script_dir / "annotations" / "instances_train2014.json"),
            "output_dir": str(script_dir / "labels" / "train2014"),
            "image_dir": str(script_dir / "images" / "train2014")
        },
        {
            "coco_json": str(script_dir / "annotations" / "instances_val2014.json"),
            "output_dir": str(script_dir / "labels" / "val2014"),
            "image_dir": str(script_dir / "images" / "val2014")
        }
    ]

    # 新增：统一生成全局类别文件（适用于多数据集共用）
    global_classes = set()
    for dataset in datasets:
        with open(dataset['coco_json'], 'r') as f:
            data = json.load(f)
            global_classes.update(cat['name'] for cat in data['categories'])
    
    # 按ID顺序排序后写入
    with open("classes.txt", 'w') as f:
        f.write('\n'.join(sorted(global_classes, key=lambda x: data['categories'][0]['id'])))

    # 批量处理数据集
    for dataset in datasets:
        print(f"\n正在处理数据集: {dataset['image_dir']}")
        coco2yolo(
            coco_json_path=dataset['coco_json'],
            output_dir=dataset['output_dir'],
            image_dir=dataset['image_dir']
        )