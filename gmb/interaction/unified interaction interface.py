class UnifiedInteraction:
    def __init__(self):
        self.manager = HardwareManager() # type: ignore
        self.manager.detect_hardware()
        
        # 如果使用模拟模式，可以加载测试视频
        if self.manager.simulation_mode:
            self.manager.camera.use_test_video("data/simulation/test_video.mp4")
    
    def get_frame(self):
        """获取当前帧"""
        return self.manager.camera.get_frame()
    
    def execute_action(self, action_type, detection=None):
        """执行动作"""
        self.manager.hand.execute_action(action_type, detection)
    
    def visualize_hardware_status(self, image):
        """可视化硬件状态"""
        if self.manager.simulation_mode:
            return self.manager.hand.visualize_status(image)
        return image
    
    def release_resources(self):
        """释放所有资源"""
        self.manager.camera.release()
        if hasattr(self.manager.hand, 'disconnect'):
            self.manager.hand.disconnect()
        print("所有硬件资源已释放")