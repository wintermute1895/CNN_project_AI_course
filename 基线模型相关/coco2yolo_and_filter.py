import os
import json
from tqdm import tqdm
from pathlib import Path
import shutil

def coco2yolo_and_filter(coco_json_path, output_label_dir, output_image_dir, image_dir, keep_classes, num_images=1000):
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建类别ID映射和名称列表（按YOLO ID顺序）
    categories = {}
    class_names = []
    for cat in coco_data['categories']:
        categories[cat['id']] = cat['name']  # 直接映射COCO ID → 类别名称
        class_names.append(cat['name'])

    # 确定保留的类别名称
    keep_class_names = set(keep_classes)
    print(f"配置保留的类别: {keep_class_names}")

    # 写入新的classes.txt（位于输出标注目录下）
    new_classes_path = os.path.join(output_label_dir, "classes.txt")
    with open(new_classes_path, 'w') as f:
        f.write('\n'.join(keep_classes))

    # 新旧类别ID映射（原始COCO ID → 新ID，从0开始）
    keep_coco_ids = [cat_id for cat_id, name in categories.items() if name in keep_class_names]
    old_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(keep_coco_ids)}
    print(f"COCO ID → 新ID映射: {old_id_to_new_id}")

    # 按图像ID分组标注
    image_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_anns:
            image_anns[image_id] = []
        image_anns[image_id].append(ann)

    kept_count = 0
    # 处理每张图像
    for img in tqdm(coco_data['images'], desc=f"Processing {os.path.basename(image_dir)}"):
        if kept_count >= num_images:
            break

        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        file_name = img['file_name']

        # 检查图像是否存在
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            print(f"警告：图像文件 {file_name} 不存在，跳过处理")
            continue

        # 获取当前图像的所有标注
        anns = image_anns.get(img_id, [])

        kept_lines = []
        for ann in anns:
            coco_id = ann['category_id']
            cat_name = categories.get(coco_id, f"未知类别_{coco_id}")
            
            # 严格过滤：只保留在keep_classes中的类别
            if cat_name not in keep_class_names:
                continue

            # 获取新ID
            new_class_id = old_id_to_new_id.get(coco_id, -1)
            if new_class_id == -1:
                print(f"警告：类别 {cat_name} (ID={coco_id}) 不在映射表中，已跳过")
                continue

            # 转换边界框
            x, y, w, h = ann['bbox']
            x_center = max(0.0, min((x + w / 2) / img_w, 1.0))
            y_center = max(0.0, min((y + h / 2) / img_h, 1.0))
            w_norm = max(0.0, min(w / img_w, 1.0))
            h_norm = max(0.0, min(h / img_h, 1.0))
            
            new_line = f"{new_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            kept_lines.append(new_line)

        # 只处理包含目标类别的图像
        if kept_lines:
            # 复制图像文件
            dst_img_path = os.path.join(output_image_dir, file_name)
            shutil.copy2(img_path, dst_img_path)

            # 写入有效标注
            txt_path = os.path.join(output_label_dir, os.path.splitext(file_name)[0] + '.txt')
            with open(txt_path, 'w') as f_txt:
                f_txt.writelines(kept_lines)

            kept_count += 1

    print(f"\n处理完成：共保留 {kept_count} 个有效标注和图像")

if __name__ == '__main__':
    # 保留的类别
    keep_classes = [
        "person", "fire hydrant",
        "chair", "potted plant", "bed", "dining table", "tv", "laptop", "mouse",
        "keyboard", "cell phone", "book", "clock", "vase",  "teddy bear"
    ]

    script_dir = Path(__file__).parent.resolve()

    # 数据集配置
    coco_json_path = str(script_dir / "annotations" / "instances_train2014.json")
    image_dir = str(script_dir / "images" / "train2014")
    output_label_dir = str(script_dir / "基线label")
    output_image_dir = str(script_dir / "基线image")

    coco2yolo_and_filter(
        coco_json_path=coco_json_path,
        output_label_dir=output_label_dir,
        output_image_dir=output_image_dir,
        image_dir=image_dir,
        keep_classes=keep_classes,
        num_images=1000
    )