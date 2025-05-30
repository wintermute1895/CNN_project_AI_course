import argparse
import time
import cv2
import torch
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import plot_one_box

# ====================== 模型输出解析模块 ======================
class DetectionParser:
    def __init__(self, model_names):
        """
        初始化检测解析器
        
        参数:
            model_names: 模型类别名称列表 (model.names)
        """
        self.names = model_names
    
    def parse_raw_detections(self, raw_detections):
        """
        解析原始检测张量为结构化数据
        
        参数:
            raw_detections: 原始检测张量 [N, 6]，格式为 [x1, y1, x2, y2, conf, cls]
            
        返回:
            结构化检测结果列表 [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class': str,
                'class_idx': int
            }]
        """
        structured_detections = []
        
        for det in raw_detections:
            *xyxy, conf, cls = det
            class_idx = int(cls.item())
            class_name = self.names[class_idx]
            
            # 将张量值转换为Python基本类型
            bbox = [int(coord.item()) for coord in xyxy]
            confidence = float(conf.item())
            
            structured_detections.append({
                "bbox": bbox,
                "confidence": confidence,
                "class": class_name,
                "class_idx": class_idx
            })
        
        return structured_detections

# ====================== 决策逻辑模块 ======================
class DecisionMaker:
    def __init__(self, critical_threshold=0.7, critical_region=(0.4, 0.6)):
        """
        初始化决策器
        
        参数:
            critical_threshold: 关键置信度阈值
            critical_region: 关键区域定义 (left_ratio, right_ratio)
        """
        self.critical_threshold = critical_threshold
        self.critical_region = critical_region
    
    def is_in_critical_region(self, bbox, img_width):
        """
        判断物体是否在关键区域
        
        参数:
            bbox: 边界框 [x1, y1, x2, y2]
            img_width: 图像宽度
            
        返回:
            bool: 是否在关键区域
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        left_bound = img_width * self.critical_region[0]
        right_bound = img_width * self.critical_region[1]
        return left_bound <= center_x <= right_bound
    
    def make_decision(self, detection, img_width):
        """
        基于检测结果做出决策
        
        参数:
            detection: 结构化检测结果
            img_width: 图像宽度
            
        返回:
            str: 决策类型 ('emergency_stop', 'avoid_object', None)
        """
        class_name = detection["class"]
        confidence = detection["confidence"]
        bbox = detection["bbox"]
        
        # 只处理高置信度的危险障碍物
        if class_name == "dangerous_obstacle" and confidence > self.critical_threshold:
            if self.is_in_critical_region(bbox, img_width):
                return "emergency_stop"
            else:
                return "avoid_object"
        
        return None

# ====================== 硬件控制模块 ======================
class HardwareController:
    @staticmethod
    def execute_action(action_type, detection=None):
        """
        执行硬件控制动作
        
        参数:
            action_type: 动作类型 ('emergency_stop', 'avoid_object')
            detection: 相关的检测结果
        """
        if action_type == "emergency_stop":
            print("执行紧急停止动作")
            # 实际硬件控制代码
            # leap_hand.send_command("STOP")
        elif action_type == "avoid_object" and detection:
            # 根据物体位置决定避让方向
            direction = "left" if detection["bbox"][0] > 0.5 else "right"
            print(f"执行避障动作: 向{direction}移动")
            # 实际硬件控制代码
            # leap_hand.send_command(f"AVOID_{direction.upper()}")
        else:
            # 默认正常操作
            print("正常操作中...")
            # leap_hand.send_command("CONTINUE")

# ====================== 可视化模块 ======================
class DetectionVisualizer:
    def __init__(self, show_center=True):
        """
        初始化可视化器
        
        参数:
            show_center: 是否显示边界框中心点
        """
        self.show_center = show_center
        self.fps = 0
    
    def visualize(self, image, detections):
        """
        在图像上可视化检测结果
        
        参数:
            image: 原始图像
            detections: 结构化检测结果列表
            
        返回:
            可视化后的图像
        """
        visualized_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            
            # 绘制边界框和标签
            label = f"{det['class']} {det['confidence']:.2f}"
            plot_one_box((x1, y1, x2, y2), visualized_image, label=label, 
                         color=(0, 255, 0), line_thickness=2)
            
            # 绘制中心点
            if self.show_center:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(visualized_image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 添加FPS信息
        cv2.putText(visualized_image, f'FPS: {self.fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return visualized_image
    
    def update_fps(self, frame_count, start_time):
        """
        更新FPS计算
        
        参数:
            frame_count: 帧计数器
            start_time: 开始时间
            
        返回:
            新的帧计数器和开始时间
        """
        elapsed = time.time() - start_time
        if elapsed > 1:  # 每秒更新一次
            self.fps = frame_count / elapsed
            return 0, time.time()  # 重置计数器和时间
        return frame_count, start_time

# ====================== 主检测类 ======================
class ObjectDetector:
    def __init__(self, weights, device='', conf_thres=0.5, iou_thres=0.45):
        # 设备设置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not device else device or 'cpu')
        
        # 加载模型
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
        self.model = self.model.to(self.device).eval()
        
        # 初始化各个模块
        self.parser = DetectionParser(self.model.names)
        self.decision_maker = DecisionMaker()
        self.visualizer = DetectionVisualizer()
        self.hardware_controller = HardwareController()
        
        # 检测参数
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 性能跟踪
        self.frame_count = 0
        self.start_time = time.time()

    def preprocess_image(self, image):
        """预处理图像"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        return img_tensor
    
    def detect(self, image):
        """执行检测并返回结构化结果"""
        # 预处理
        img_tensor = self.preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        
        # NMS处理
        raw_detections = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # 处理第一张图像的检测结果
        if len(raw_detections) > 0:
            return self.parser.parse_raw_detections(raw_detections[0])
        return []
    
    def process_frame(self, image):
        """完整处理一帧图像：检测、决策、可视化"""
        # 执行检测
        detections = self.detect(image)
        
        # 更新FPS
        self.frame_count, self.start_time = self.visualizer.update_fps(
            self.frame_count + 1, self.start_time
        )
        
        # 处理检测结果并做决策
        img_width = image.shape[1]
        for detection in detections:
            action = self.decision_maker.make_decision(detection, img_width)
            if action:
                self.hardware_controller.execute_action(action, detection)
        
        # 可视化结果
        return self.visualizer.visualize(image, detections), detections

# ====================== 检测函数 ======================
def detect_image(weights, source, output, conf_thres=0.5, iou_thres=0.45, device=''):
    """图像检测函数"""
    # 初始化检测器
    detector = ObjectDetector(weights, device, conf_thres, iou_thres)
    
    # 加载图像
    img = cv2.imread(source)
    if img is None:
        print(f"错误：无法加载图像 {source}")
        return
    
    # 处理图像
    result_img, detections = detector.process_frame(img)
    
    # 打印检测结果
    print(f"\n检测结果 ({source}):")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class']} (置信度: {det['confidence']:.2f})")
        print(f"     位置: [{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]")
    
    # 保存结果
    cv2.imwrite(output, result_img)
    print(f"\n结果已保存至: {output}")
    
    # 显示结果
    cv2.imshow('Detection Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video(weights, source, output, conf_thres=0.5, iou_thres=0.45, device=''):
    """视频检测函数"""
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
        
        # 处理当前帧
        result_frame, _ = detector.process_frame(frame)
        
        # 写入输出视频
        out.write(result_frame)
        
        # 显示实时画面
        cv2.imshow('Real-time Detection', result_frame)
        
        # 退出检测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频处理完成，保存至: {output}")

# ====================== 主函数 ======================
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