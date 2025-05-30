import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        print("RealSense相机已启动")
    
    def get_frame(self):
        """获取当前帧"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        
        # 转换为OpenCV格式
        frame = np.asanyarray(color_frame.get_data())
        return frame
    
    def release(self):
        """释放资源"""
        self.pipeline.stop()
        print("RealSense相机已释放")
        