import cv2
import numpy as np
import os
import random
from PIL import Image, ImageEnhance
import xml.etree.ElementTree as ET
import math

def cv_imread(file_path):
    """支持中文路径的OpenCV图像读取函数"""
    img_data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    return img

def cv_imwrite(file_path, image):
    """支持中文路径的OpenCV图像保存函数"""
    ext = os.path.splitext(file_path)[1]
    success, encoded_image = cv2.imencode(ext, image)
    if success:
        with open(file_path, 'wb') as f:
            f.write(encoded_image)
    return success

def rotate_image_and_bbox(image, bboxes, angle):
    """
    旋转图像和边界框
    :param image: 输入图像 (OpenCV格式)
    :param bboxes: YOLO格式边界框列表 [[class_id, x_center, y_center, width, height], ...]
    :param angle: 旋转角度（度）
    :return: 旋转后的图像和边界框
    """
    img_h, img_w = image.shape[:2]  # 重命名避免变量冲突
    center = (img_w // 2, img_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # 计算新图像尺寸
    nW = int((img_h * sin) + (img_w * cos))
    nH = int((img_h * cos) + (img_w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    
    # 旋转图像
    rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # 旋转边界框（修复坐标转换错误）
    rotated_bboxes = []
    for bbox in bboxes:
        class_id, x, y, w, h = bbox
        
        # 转换为绝对坐标（使用图像尺寸，而非边界框尺寸）
        abs_x = x * img_w
        abs_y = y * img_h
        abs_w = w * img_w
        abs_h = h * img_h
        
        # 计算边界框四个角点
        points = np.array([
            [abs_x - abs_w/2, abs_y - abs_h/2],
            [abs_x + abs_w/2, abs_y - abs_h/2],
            [abs_x + abs_w/2, abs_y + abs_h/2],
            [abs_x - abs_w/2, abs_y + abs_h/2]
        ])
        
        # 应用旋转矩阵
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_points = M.dot(points_ones.T).T
        
        # 计算新边界框
        min_x, min_y = np.min(transformed_points, axis=0)
        max_x, max_y = np.max(transformed_points, axis=0)
        
        # 转换为YOLO格式
        new_x = ((min_x + max_x) / 2) / nW
        new_y = ((min_y + max_y) / 2) / nH
        new_w = (max_x - min_x) / nW
        new_h = (max_y - min_y) / nH
        
        # 确保边界框在图像内且大小合理
        if (0 <= new_x <= 1 and 0 <= new_y <= 1 and 
            0.01 <= new_w <= 0.99 and 0.01 <= new_h <= 0.99):
            rotated_bboxes.append([class_id, new_x, new_y, new_w, new_h])
    
    return rotated, rotated_bboxes

def adjust_exposure(image, factor):
    """
    调整图像曝光度
    :param image: 输入图像 (OpenCV格式)
    :param factor: 曝光调整因子 (0.5=变暗, 1.0=不变, 2.0=变亮)
    :return: 调整后的图像
    """
    # 使用PIL进行更自然的曝光调整
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    adjusted = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(adjusted), cv2.COLOR_RGB2BGR)

def adjust_hue(image, factor):
    """
    调整图像色调
    :param image: 输入图像 (OpenCV格式)
    :param factor: 色调调整因子 (-1.0到1.0)
    :return: 调整后的图像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + factor * 180) % 180
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_crop(image, bboxes, min_scale=0.6, max_scale=0.9):
    """
    随机裁切图像并调整边界框
    :param image: 输入图像
    :param bboxes: YOLO格式边界框列表
    :param min_scale: 最小裁切比例
    :param max_scale: 最大裁切比例
    :return: 裁切后的图像和边界框
    """
    h, w = image.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 随机选择裁切起点
    x = random.randint(0, w - new_w)
    y = random.randint(0, h - new_h)
    
    # 裁切图像
    cropped = image[y:y+new_h, x:x+new_w]
    
    # 调整边界框（修复坐标转换错误）
    cropped_bboxes = []
    for bbox in bboxes:
        class_id, bx, by, bw, bh = bbox
        
        # 转换为绝对坐标（使用图像尺寸）
        abs_x = bx * w
        abs_y = by * h
        abs_w = bw * w
        abs_h = bh * h
        
        # 计算边界框在新图像中的位置
        new_x = (abs_x - x) / new_w
        new_y = (abs_y - y) / new_h
        new_w_val = abs_w / new_w
        new_h_val = abs_h / new_h
        
        # 检查边界框是否在裁切区域内
        if (0 <= new_x <= 1 and 0 <= new_y <= 1 and 
            new_x - new_w_val/2 >= 0 and new_x + new_w_val/2 <= 1 and
            new_y - new_h_val/2 >= 0 and new_y + new_h_val/2 <= 1):
            cropped_bboxes.append([class_id, new_x, new_y, new_w_val, new_h_val])
    
    return cropped, cropped_bboxes

def read_yolo_annotation(annotation_path):
    """
    读取YOLO格式的标注文件
    :param annotation_path: 标注文件路径
    :return: 边界框列表 [[class_id, x_center, y_center, width, height], ...]
    """
    bboxes = []
    if not os.path.exists(annotation_path):
        print(f"⚠️ 标注文件不存在: {annotation_path}")
        return bboxes
        
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip().split()
            if len(data) == 5:
                try:
                    class_id = int(data[0])
                    x_center = float(data[1])
                    y_center = float(data[2])
                    width = float(data[3])
                    height = float(data[4])
                    bboxes.append([class_id, x_center, y_center, width, height])
                except ValueError:
                    print(f"⚠️ 标注格式错误: {line.strip()}")
    return bboxes

def write_yolo_annotation(annotation_path, bboxes):
    """
    写入YOLO格式的标注文件
    :param annotation_path: 输出标注文件路径
    :param bboxes: 边界框列表
    """
    with open(annotation_path, 'w', encoding='utf-8') as f:
        for bbox in bboxes:
            line = f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n"
            f.write(line)

def process_dataset(input_dir, output_dir, operations):
    """
    处理整个数据集（添加中文路径支持）
    :param input_dir: 输入数据集目录
    :param output_dir: 输出数据集目录
    :param operations: 要应用的操作列表
    """
    # 创建输出目录（添加exist_ok避免错误）
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # 遍历所有图像文件（添加路径验证）
    image_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')
    
    if not os.path.exists(image_dir):
        print(f"❌ 图像目录不存在: {image_dir}")
        return
    if not os.path.exists(label_dir):
        print(f"❌ 标注目录不存在: {label_dir}")
        return
    
    processed_count = 0
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            base_name = os.path.splitext(image_name)[0]
            label_path = os.path.join(label_dir, base_name + '.txt')
            
            # 使用支持中文路径的读取方法
            image = cv_imread(image_path)
            if image is None:
                print(f"⚠️ 无法读取图像: {image_path}")
                continue
                
            # 添加标注文件存在性检查
            if not os.path.exists(label_path):
                print(f"⚠️ 标注文件不存在: {label_path}")
                continue
                
            bboxes = read_yolo_annotation(label_path)
            if not bboxes:
                print(f"⚠️ 无有效标注: {label_path}")
                continue
                
            # 应用所有操作
            processed_image = image.copy()
            processed_bboxes = [bbox.copy() for bbox in bboxes]
            
            for operation in operations:
                try:
                    if operation == 'rotate':
                        angle = random.uniform(-30, 30)
                        processed_image, processed_bboxes = rotate_image_and_bbox(
                            processed_image, processed_bboxes, angle)
                    
                    elif operation == 'exposure':
                        factor = random.uniform(0.7, 1.3)
                        processed_image = adjust_exposure(processed_image, factor)
                    
                    elif operation == 'hue':
                        factor = random.uniform(-0.1, 0.1)
                        processed_image = adjust_hue(processed_image, factor)
                    
                    elif operation == 'crop':
                        processed_image, processed_bboxes = random_crop(
                            processed_image, processed_bboxes)
                except Exception as e:
                    print(f"⚠️ 操作 {operation} 失败: {e}")
                    continue
            
            # 保存处理后的图像和标注
            output_image_path = os.path.join(output_dir, 'images', image_name)
            output_label_path = os.path.join(output_dir, 'labels', base_name + '.txt')
            
            # 使用支持中文路径的保存方法
            if not cv_imwrite(output_image_path, processed_image):
                print(f"❌ 图像保存失败: {output_image_path}")
            write_yolo_annotation(output_label_path, processed_bboxes)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"✅ 已处理 {processed_count} 张图像")

# 使用示例
if __name__ == "__main__":
    # 配置参数（使用原始字符串避免转义问题）
    INPUT_DATASET = r"E:\vs doc\change\自己的数据集"  # 输入数据集路径
    OUTPUT_DATASET = r"E:\vs doc\change\输出数据集"  # 输出数据集路径
    OPERATIONS = ['rotate', 'exposure', 'hue']  # 要应用的操作'rotate', 'exposure', 'hue', 'crop'
    
    # 处理数据集（添加异常捕获）
    try:
        print("🚀 开始处理数据集...")
        process_dataset(
            input_dir=INPUT_DATASET,
            output_dir=OUTPUT_DATASET,
            operations=OPERATIONS
        )
        print("✅ 数据集处理完成")
    except Exception as e:
        print(f"❌ 处理失败: {e}")