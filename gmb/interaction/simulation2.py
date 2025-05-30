import time

class SimulatedHand:
    def __init__(self):
        self.current_action = "NORMAL"
        self.action_history = []
        print("模拟灵巧手已启动")
    
    def execute_action(self, action_type, detection=None):
        """执行模拟动作"""
        action_time = time.strftime("%H:%M:%S")
        
        if action_type == "emergency_stop":
            self.current_action = "EMERGENCY_STOP"
            action_desc = "紧急停止"
        elif action_type == "avoid_object":
            self.current_action = "AVOID"
            direction = "左" if detection and detection["bbox"][0] > 0.5 else "右"
            action_desc = f"向{direction}避让"
        else:
            self.current_action = "NORMAL"
            action_desc = "正常操作"
        
        # 记录动作历史
        self.action_history.append({
            "time": action_time,
            "action": action_desc,
            "detection": detection
        })
        
        print(f"[模拟动作] {action_time}: {action_desc}")
    
    def get_status(self):
        """获取当前状态"""
        return {
            "current_action": self.current_action,
            "action_history": self.action_history[-5:]  # 返回最近5个动作
        }
    
    def visualize_status(self, image):
        """在图像上可视化状态"""
        # 添加状态文本
        status_text = f"状态: {self.current_action}"
        cv2.putText(image, status_text, (10, image.shape[0] - 30),  # type: ignore
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # type: ignore
        
        # 添加最近动作
        if self.action_history:
            last_action = self.action_history[-1]
            action_text = f"上次动作: {last_action['time']} - {last_action['action']}"
            cv2.putText(image, action_text, (10, image.shape[0] - 60),  # type: ignore
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # type: ignore
        
        return image