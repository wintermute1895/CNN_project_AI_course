class HardwareManager:
    def __init__(self):
        self.camera = None
        self.hand = None
        self.hardware_available = False
        self.simulation_mode = False
        
    def detect_hardware(self):
        """检测可用硬件"""
        # 检测相机
        camera_detected = self._detect_camera()
        
        # 检测灵巧手
        hand_detected = self._detect_leap_hand()
        
        # 设置硬件状态
        self.hardware_available = camera_detected and hand_detected
        self.simulation_mode = not self.hardware_available
        
        if self.hardware_available:
            print("检测到硬件设备，使用真实硬件模式")
            self._init_real_hardware()
        else:
            print("未检测到硬件设备，使用模拟模式")
            self._init_simulation()
    
    def _detect_camera(self):
        """检测D405相机"""
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) > 0:
                print(f"检测到RealSense设备: {devices[0].get_info(rs.camera_info.name)}")
                return True
        except ImportError:
            print("pyrealsense2未安装，无法使用真实相机")
        return False
    
    def _detect_leap_hand(self):
        """检测LEAP HAND灵巧手"""
        try:
            # 这里使用厂商提供的检测方法
            # 实际项目中替换为真实检测逻辑
            from leap_hand_sdk import LeapHand # type: ignore
            if LeapHand.is_connected():
                print("检测到LEAP HAND灵巧手")
                return True
        except ImportError:
            print("leap_hand_sdk未安装，无法使用真实灵巧手")
        return False
    
    def _init_real_hardware(self):
        """初始化真实硬件"""
        self.camera = RealSenseCamera() # type: ignore
        self.hand = LeapHandController() # type: ignore
    
    def _init_simulation(self):
        """初始化模拟系统"""
        self.camera = SimulatedCamera() # type: ignore
        self.hand = SimulatedHand() # type: ignore