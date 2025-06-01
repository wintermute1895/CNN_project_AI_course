import sys
import os

# 修复 Pillow 路径问题
venv_path = r"D:\12345\yolov5-env"
site_packages = os.path.join(venv_path, "Lib", "site-packages")
if site_packages not in sys.path:
    sys.path.append(site_packages)

# 确保 Pillow 已安装
try:
    from PIL import Image
    print(f"Pillow 已安装，版本: {Image.__version__}")
except ImportError:
    print("正在安装 Pillow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    from PIL import Image
    import cv2
import random

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    在图像上绘制一个边界框
    
    参数:
    x: 边界框坐标 [x1, y1, x2, y2]
    img: 要绘制框的图像 (numpy数组)
    color: BGR颜色元组
    label: 标签文本
    line_thickness: 线宽
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        import sys
import os

# 添加项目根目录到 Python 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'yolov5'))
import sys
print("Python 路径:", sys.executable)
print("Pillow 路径:", [p for p in sys.path if 'PIL' in p or 'pillow' in p])
import sys
print("=== Python 模块搜索路径 ===")
for path in sys.path:
    print(path)
print("=========================")
import argparse
import time
import cv2
import torch
import numpy as np
from yolov5.utils.general import non_max_suppression
# 不再从外部导入，使用我们自定义的函数

# 硬件控制模拟（实际项目中替换为真实硬件接口）
class HardwareControl:
    @staticmethod
    def emergency_stop():
        print("执行紧急停止动作")
        # 实际硬件控制代码
        # leap_hand.send_command("STOP")
    
    @staticmethod
    def avoid_object(direction="left"):
        print(f"执行避障动作: 向{direction}移动")
        # 实际硬件控制代码
        # leap_hand.send_command(f"AVOID_{direction.upper()}")
    
    @staticmethod
    def normal_operation():
        print("正常操作中...")
        # leap_hand.send_command("CONTINUE")

# 检测结果处理类
class DetectionProcessor:
    def __init__(self, model, names):
        self.model = model
        self.names = names
    
    def parse_detections(self, detections):
        """解析检测结果"""
        results = []
        for det in detections:
            *xyxy, conf, cls = det
            class_idx = int(cls.item())
            class_name = self.names[class_idx]
            bbox = [int(coord.item()) for coord in xyxy]
            confidence = float(conf.item())
            
            results.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "class_idx": class_idx
            })
        return results
    
    def visualize_detections(self, image, detections, show_center=True):
        """可视化检测结果"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            
            # 绘制边界框和标签
            label = f"{det['class']} {det['confidence']:.2f}"
            plot_one_box((x1, y1, x2, y2), image, label=label, color=(0, 255, 0), line_thickness=2)
            
            # 绘制中心点
            if show_center:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 添加FPS信息
        if hasattr(self, 'fps'):
            cv2.putText(image, f'FPS: {self.fps:.1f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image
    
    def is_critical_region(self, bbox, img_width):
        """判断物体是否在关键区域（中央区域）"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        # 定义中央区域（屏幕宽度40%-60%）
        return img_width * 0.4 <= center_x <= img_width * 0.6
    
    def make_decision(self, detection, img_width):
        """基于检测结果做出决策"""
        class_name = detection["class"]
        confidence = detection["confidence"]
        bbox = detection["bbox"]
        
        # 决策逻辑
        if class_name == "dangerous_obstacle" and confidence > 0.7:
            if self.is_critical_region(bbox, img_width):
                print("警告：关键障碍物在中央区域！立即停止")
                HardwareControl.emergency_stop()
                return "emergency_stop"
            else:
                print("警告：检测到危险障碍物，建议避让")
                HardwareControl.avoid_object("left" if bbox[0] > img_width/2 else "right")
                return "avoid_object"
        return None

# 主检测类
class ObjectDetector:
    def __init__(self, weights, device='', conf_thres=0.5, iou_thres=0.45):
        # 设备设置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not device else device or 'cpu')
        
        # 加载模型
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
        self.model = self.model.to(self.device).eval()
        
        # 初始化处理器
        self.processor = DetectionProcessor(self.model, self.model.names)
        
        # 检测参数
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 性能跟踪
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
    def preprocess_image(self, image):
        """预处理图像"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        return img_tensor
    
    def detect(self, image):
        """执行检测"""
        # 预处理
        img_tensor = self.preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        
        # NMS处理
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # 处理第一张图像的检测结果
        detections = pred[0] if len(pred) > 0 else []
        
        # 转换为结构化数据
        structured_detections = self.processor.parse_detections(detections) if len(detections) > 0 else []
        
        # 更新FPS
        self.update_fps()
        
        return structured_detections
    
    def update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1:  # 每秒更新一次
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
            self.processor.fps = self.fps  # 传递给处理器用于显示

# 图像检测函数
def detect_image(weights, source, output, conf_thres=0.5, iou_thres=0.45, device=''):
    # 初始化检测器
    detector = ObjectDetector(weights, device, conf_thres, iou_thres)
    
    # 加载图像
    img = cv2.imread(source)
    if img is None:
        print(f"错误：无法加载图像 {source}")
        return
    
    # 执行检测
    detections = detector.detect(img)
    
    # 处理检测结果
    img_width = img.shape[1]
    for detection in detections:
        # 决策
        detector.processor.make_decision(detection, img_width)
        
        # 打印信息
        print(f"检测到: {detection['class']} | 置信度: {detection['confidence']:.2f}")
        print(f"位置: 左上({detection['bbox'][0]},{detection['bbox'][1]}) 右下({detection['bbox'][2]},{detection['bbox'][3]})")
    
    # 可视化结果
    img = detector.processor.visualize_detections(img, detections)
    
    # 保存结果
    cv2.imwrite(output, img)
    print(f"结果已保存至: {output}")
    
    # 显示结果
    cv2.imshow('Detection Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 视频检测函数
def detect_video(weights, source, output, conf_thres=0.5, iou_thres=0.45, device=''):
    # 初始化检测器
    detector = ObjectDetector(weights, device, conf_thres, iou_thres)
    
    # 初始化视频源
    is_camera = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_camera else source)
    if not cap.isOpened():
        print(f"错误：无法打开视频源 {source}")
        return
    
    # 准备视频输出
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))
    
    print("开始实时检测，按 'q' 键退出...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 执行检测
        detections = detector.detect(frame)
        
        # 处理检测结果并做决策
        for detection in detections:
            detector.processor.make_decision(detection, frame_width)
        
        # 可视化结果
        frame = detector.processor.visualize_detections(frame, detections)
        
        # 写入输出视频
        out.write(frame)
        
        # 显示实时画面
        cv2.imshow('Real-time Detection', frame)
        
        # 退出检测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频处理完成，保存至: {output}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description='YOLOv5物体检测器')
    subparsers = parser.add_subparsers(dest='command', help='检测模式')
    
    # 图像检测参数
    image_parser = subparsers.add_parser('image', help='图像检测模式')
    image_parser.add_argument('--weights', type=str, default='best.pt', help='模型权重路径')
    image_parser.add_argument('--source', type=str, required=True, help='源图像路径')
    image_parser.add_argument('--output', type=str, default='detection_result.jpg', help='输出图像路径')
    image_parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    image_parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU阈值')
    image_parser.add_argument('--device', default='', help='设备: cuda:0 或 cpu')
    
    # 视频检测参数
    video_parser = subparsers.add_parser('video', help='视频检测模式')
    video_parser.add_argument('--weights', type=str, default='best.pt', help='模型权重路径')
    video_parser.add_argument('--source', type=str, required=True, help='视频源: 文件路径或摄像头ID (0)')
    video_parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    video_parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    video_parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU阈值')
    video_parser.add_argument('--device', default='', help='设备: cuda:0 或 cpu')
    
    args = parser.parse_args()
    
    if args.command == 'image':
        detect_image(
            weights=args.weights,
            source=args.source,
            output=args.output,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            device=args.device
        )
    elif args.command == 'video':
        detect_video(
            weights=args.weights,
            source=args.source,
            output=args.output,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            device=args.device
        )
    else:
        print("请指定检测模式: 'image' 或 'video'")
        print("示例:")
        print("  检测图像: python detector.py image --source data/images/test.jpg")
        print("  检测视频: python detector.py video --source data/videos/test.mp4")
        print("  使用摄像头: python detector.py video --source 0")

if __name__ == "__main__":
    main()