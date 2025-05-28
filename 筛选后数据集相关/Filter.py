import os
import shutil
from tqdm import tqdm
from pathlib import Path

def filter_yolo_dataset(input_labels_dir, output_labels_dir, keep_classes, 
                        input_images_dir=None, output_images_dir=None, 
                        remove_empty_images=True):
    """
    过滤YOLO格式数据集，排除对classes.txt的处理，并修复标注解析逻辑
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    if output_images_dir:
        os.makedirs(output_images_dir, exist_ok=True)
    
    # 从标注目录内读取classes.txt（确保路径正确）
    classes_path = os.path.join(input_labels_dir, "classes.txt")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"未找到类别文件: {classes_path}")
    
    with open(classes_path, 'r') as f:
        original_classes = [line.strip() for line in f.readlines()]
    print(f"原始类别列表（{input_labels_dir}）: {original_classes}")
    
    # 确定保留的类别ID和名称
    keep_class_ids = [idx for idx, name in enumerate(original_classes) if name in keep_classes]
    new_classes = [original_classes[idx] for idx in keep_class_ids]
    
    if not new_classes:
        print("警告：未找到需要保留的类别，处理终止")
        return
    
    print(f"将保留的类别（ID映射）: {dict(zip(new_classes, keep_class_ids))}")
    
    # 写入新的classes.txt（位于输出标注目录的父级或同级）
    output_parent_dir = os.path.dirname(output_labels_dir)
    new_classes_path = os.path.join(output_parent_dir, "classes.txt")
    with open(new_classes_path, 'w') as f:
        f.write('\n'.join(new_classes))
    
    # 新旧类别ID映射（原始ID → 新ID，从0开始）
    old_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(keep_class_ids)}
    
    # **关键修复：排除classes.txt文件，仅处理标注文件**
    label_files = [
        f for f in os.listdir(input_labels_dir) 
        if f.endswith('.txt') and f != "classes.txt"  # 排除类别文件
    ]
    total_files = len(label_files)
    kept_files = 0
    empty_files = 0

    for label_file in tqdm(label_files, desc="处理标注文件"):
        input_path = os.path.join(input_labels_dir, label_file)
        output_path = os.path.join(output_labels_dir, label_file)
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        kept_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            try:
                old_class_id = int(parts[0])  # 确保第一列是整数ID
            except ValueError:
                print(f"警告：标注文件 {label_file} 格式错误，第一列非整数: {parts[0]}，已跳过")
                continue
            
            if old_class_id in keep_class_ids:
                new_class_id = old_id_to_new_id[old_class_id]
                new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                kept_lines.append(new_line)
        
        # 写入有效标注或删除空文件
        if kept_lines:
            with open(output_path, 'w') as f:
                f.writelines(kept_lines)
            kept_files += 1
            
            # 复制图像文件
            if input_images_dir and output_images_dir:
                image_name = os.path.splitext(label_file)[0]
                found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    src_path = os.path.join(input_images_dir, f"{image_name}{ext}")
                    dst_path = os.path.join(output_images_dir, f"{image_name}{ext}")
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                        found = True
                        break
                if not found:
                    print(f"警告：未找到图像文件 for {label_file}")
        else:
            empty_files += 1
            if remove_empty_images:
                try:
                    os.remove(output_path)
                except FileNotFoundError:
                    pass  # 防止文件已被删除或不存在
    
    print(f"\n处理完成：共 {total_files} 个标注文件，保留 {kept_files} 个有效标注，移除 {empty_files} 个空标注")
    if input_images_dir and output_images_dir:
        print(f"图像已复制到 {output_images_dir}")

if __name__ == '__main__':
    # 保留的类别（需与classes.txt中的名称完全一致）
    keep_classes = [
        "person", "fire hydrant", 
        "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", 
        "book", "clock", "vase",  "teddy bear"
    ]
    
    script_dir = Path(__file__).parent.resolve()
    
    # 数据集配置（示例路径，需根据实际情况修改）
    datasets = [
        {
            "input_labels": "labels/train2014",
            "input_images": "images/train2014",
            "output_labels": "labels_filtered/train2014",
            "output_images": "images_filtered/train2014"
        },
        {
            "input_labels": "labels/val2014",
            "input_images": "images/val2014",
            "output_labels": "labels_filtered/val2014",
            "output_images": "images_filtered/val2014"
        }
    ]
    
    for dataset in datasets:
        print(f"\n开始处理 {dataset['input_labels']}...")
        filter_yolo_dataset(
            input_labels_dir=str(script_dir / dataset["input_labels"]),
            output_labels_dir=str(script_dir / dataset["output_labels"]),
            keep_classes=keep_classes,
            input_images_dir=str(script_dir / dataset["input_images"]),
            output_images_dir=str(script_dir / dataset["output_images"]),
            remove_empty_images=True
        )