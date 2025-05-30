import cv2
import random

class SimulatedCamera:
    def __init__(self):
        # 使用测试图像或视频
        self.test_images = self._load_test_images()
        self.current_index = 0
        self.video_source = None
        self.use_video = False
        print("模拟相机已启动")
    
    def _load_test_images(self):
        """加载测试图像"""
        images = []
        for i in range(1, 11):
            try:
                img = cv2.imread(f"data/simulation/test_{i}.jpg")
                if img is not None:
                    images.append(img)
            except:
                pass
        return images
    
    def use_test_video(self, video_path):
        """使用测试视频作为源"""
        self.video_source = cv2.VideoCapture(video_path)
        self.use_video = True
        print(f"使用测试视频: {video_path}")
    
    def get_frame(self):
        """获取模拟帧"""
        if self.use_video and self.video_source:
            ret, frame = self.video_source.read()
            if not ret:
                # 循环播放
                self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_source.read()
            return frame
        
        # 使用测试图像
        if self.test_images:
            frame = self.test_images[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.test_images)
            return frame
        
        # 生成随机图像作为后备
        return self._generate_random_image()
    
    def _generate_random_image(self, width=640, height=480):
        """生成随机图像"""
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8) # type: ignore
        
        # 添加一些随机形状模拟环境
        color = random.randint(0, 255), (random.randint(0, 255)), (random.randint(0, 255))
        center = (random.randint(100, width-100), random.randint(100, height-100))
        radius = random.randint(30, 100)
        cv2.circle(img, center, radius, color, -1)
        
        return img
    
    def release(self):
        """释放资源"""
        if self.video_source:
            self.video_source.release()
        print("模拟相机已释放")