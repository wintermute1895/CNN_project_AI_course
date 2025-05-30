class ObjectDetector:
    def __init__(self, weights, device='', conf_thres=0.5, iou_thres=0.45):
        # ...（之前的初始化代码）
        
        # 添加硬件交互
        self.interaction = UnifiedInteraction() # type: ignore
    
    def process_frame(self, image):
        """完整处理一帧图像：检测、决策、交互、可视化"""
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
                # 执行硬件/模拟动作
                self.interaction.execute_action(action, detection)
        
        # 可视化结果
        result_img = self.visualizer.visualize(image, detections)
        
        # 添加硬件状态信息
        result_img = self.interaction.visualize_hardware_status(result_img)
        
        return result_img, detections
    
    def run_real_time_detection(self, source='camera'):
        """运行实时检测"""
        if source == 'camera':
            while True:
                frame = self.interaction.get_frame()
                if frame is None:
                    continue
                
                result_frame, _ = self.process_frame(frame)
                
                cv2.imshow('Real-time Detection', result_frame) # type: ignore
                if cv2.waitKey(1) & 0xFF == ord('q'): # type: ignore
                    break
        else:
            # 处理视频文件
            cap = cv2.VideoCapture(source) # type: ignore
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                result_frame, _ = self.process_frame(frame)
                
                cv2.imshow('Video Detection', result_frame) # type: ignore
                if cv2.waitKey(1) & 0xFF == ord('q'): # type: ignore
                    break
            
            cap.release()
        
        cv2.destroyAllWindows() # type: ignore
        self.interaction.release_resources()