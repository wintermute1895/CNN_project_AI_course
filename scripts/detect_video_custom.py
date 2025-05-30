import argparse
import time
import cv2
import torch
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import plot_one_box

def main():
    # 1. 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, default='0', help='视频源: 文件路径或摄像头ID (0)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--device', default='', help='设备: cuda:0 或 cpu')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    opt = parser.parse_args()
    
    # 2. 设备设置
    device = torch.device('cuda:0' if torch.cuda.is_available() and not opt.device else opt.device or 'cpu')
    
    # 3. 加载模型
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.weights)
    model = model.to(device).eval()
    
    # 4. 初始化视频源
    is_camera = opt.source.isdigit()
    cap = cv2.VideoCapture(int(opt.source) if is_camera else cv2.VideoCapture(opt.source)
    assert cap.isOpened(), f'无法打开视频源: {opt.source}'
    
    # 5. 准备视频输出
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(opt.output, fourcc, fps, (frame_width, frame_height))
    
    # 6. 实时检测循环
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 7. 预处理
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        
        # 8. 推理
        with torch.no_grad():
            pred = model(img_tensor)[0]
        
        # 9. NMS处理
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
        
        # 10. 处理检测结果
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    class_name = model.names[int(cls)]
                    
                    # 绘制边界框
                    label = f'{class_name} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)
                    
                    # 决策逻辑（示例）
                    if class_name == "dangerous_obstacle" and conf > 0.7:
                        print(f"帧 {frame_count}: 检测到关键障碍物 {class_name}!")
                        # 触发行动（硬件或模拟）
                        # take_action("avoid")
        
        # 11. 显示FPS
        fps_text = f'FPS: {1/(time.time()-start_time):.1f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        start_time = time.time()
        
        # 12. 显示并保存结果
        cv2.imshow('Real-time Detection', frame)
        out.write(frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # 13. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频处理完成，保存至: {opt.output}")

if __name__ == "__main__":
    main()