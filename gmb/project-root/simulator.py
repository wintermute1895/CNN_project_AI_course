import cv2
import numpy as np
import os

def generate_test_images(output_dir, num_images=10, size=(640, 480)):
    """生成测试图像"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(1, num_images+1):
        # 创建随机背景
        img = np.random.randint(50, 200, (size[1], size[0], 3), dtype=np.uint8)
        
        # 添加随机形状
        shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        
        if shape_type == 'circle':
            center = (np.random.randint(100, size[0]-100), np.random.randint(100, size[1]-100))
            radius = np.random.randint(30, 100)
            cv2.circle(img, center, radius, color, -1)
            label = f"Circle {i}"
        elif shape_type == 'rectangle':
            pt1 = (np.random.randint(50, size[0]-150), np.random.randint(50, size[1]-150))
            pt2 = (pt1[0]+np.random.randint(50, 150), pt1[1]+np.random.randint(50, 150))
            cv2.rectangle(img, pt1, pt2, color, -1)
            label = f"Rectangle {i}"
        else:  # triangle
            pts = np.array([
                [np.random.randint(50, size[0]-50), np.random.randint(50, size[1]-150)],
                [np.random.randint(50, size[0]-150), np.random.randint(150, size[1]-50)],
                [np.random.randint(150, size[0]-50), np.random.randint(150, size[1]-50)]
            ])
            cv2.fillPoly(img, [pts], color)
            label = f"Triangle {i}"
        
        # 添加标签
        cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 保存图像
        cv2.imwrite(os.path.join(output_dir, f"test_{i}.jpg"), img)
        print(f"生成测试图像: test_{i}.jpg")

def generate_test_video(output_path, duration=10, fps=30, size=(640, 480)):
    """生成测试视频"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    num_frames = duration * fps
    for i in range(num_frames):
        # 创建渐变背景
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (size[0], size[1]), 
                     (int(255*i/num_frames), int(128*(1+np.sin(i/10))), 200), -1)
        
        # 添加移动物体
        x = int(size[0] * i / num_frames)
        y = size[1] // 2 + int(100 * np.sin(i/20))
        cv2.circle(img, (x, y), 30, (0, 0, 255), -1)
        
        # 添加帧编号
        cv2.putText(img, f"Frame: {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(img)
    
    out.release()
    print(f"生成测试视频: {output_path}")

if __name__ == "__main__":
    # 生成测试数据
    sim_dir = "data/simulation"
    generate_test_images(sim_dir)
    generate_test_video(os.path.join(sim_dir, "test_video.mp4"))