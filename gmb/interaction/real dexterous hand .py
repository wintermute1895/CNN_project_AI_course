class LeapHandController:
    def __init__(self):
        # 初始化与LEAP HAND的连接
        from leap_hand_sdk import LeapHand # type: ignore
        self.hand = LeapHand()
        self.hand.connect()
        print("LEAP HAND灵巧手已连接")
    
    def execute_action(self, action_type, detection=None):
        """执行动作"""
        if action_type == "emergency_stop":
            self._emergency_stop()
        elif action_type == "avoid_object":
            self._avoid_object(detection)
        else:
            self._normal_operation()
    
    def _emergency_stop(self):
        """紧急停止"""
        # 发送停止命令
        self.hand.send_command("EMERGENCY_STOP")
        print("执行紧急停止动作")
    
    def _avoid_object(self, detection):
        """避让物体"""
        if detection:
            # 根据物体位置决定避让方向
            direction = "LEFT" if detection["bbox"][0] > 0.5 else "RIGHT"
            self.hand.send_command(f"AVOID_{direction}")
            print(f"执行避障动作: 向{direction}移动")
        else:
            self.hand.send_command("AVOID_LEFT")
            print("执行避障动作: 向左移动")
    
    def _normal_operation(self):
        """正常操作"""
        self.hand.send_command("NORMAL_OPERATION")
        print("正常操作中...")
    
    def disconnect(self):
        """断开连接"""
        self.hand.disconnect()
        print("LEAP HAND灵巧手已断开")