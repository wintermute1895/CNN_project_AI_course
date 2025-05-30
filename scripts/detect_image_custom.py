import argparse
import cv2
import torch
import numpy as np
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import plot_one_box

def main():
    # 1. 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, default='data/images/test.jpg', help='测试图片路径')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--device', default='', help='设备: cuda:0 或 cpu')
    opt = parser.parse_args()
    
    # 2. 设备设置
    device = torch.device('cuda:0' if torch.cuda.is_available() and not opt.device else opt.device or 'cpu')
    
    # 3. 加载模型
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.weights)
    model = model.to(device).eval()
    
    # 4. 加载图像
    img = cv2.imread(opt.source)
    assert img is not None, f'图像未找到: {opt.source}'
    
    # 5. 预处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    
    # 6. 推理
    with torch.no_grad():
        pred = model(img_tensor)[0]
    
    # 7. NMS处理
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
    
    # 8. 处理检测结果
    detections = []
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in det:
                class_name = model.names[int(cls)]
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x) for x in xyxy]
                })
                
                # 9. 绘制边界框
                label = f'{class_name} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=(0, 255, 0), line_thickness=2)
                
                # 10. 决策逻辑（示例）
                if class_name == "dangerous_obstacle" and conf > 0.7:
                    print(f"警告：检测到关键障碍物 {class_name}！置信度: {conf:.2f}")
                    # 触发行动（硬件或模拟）
                    # take_action("avoid")
    
    # 11. 保存并显示结果
    output_path = f"results/{opt.source.split('/')[-1]}"
    cv2.imwrite(output_path, img)
    print(f"结果已保存至: {output_path}")
    
    # 显示结果（可选）
    cv2.imshow('Detection Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()