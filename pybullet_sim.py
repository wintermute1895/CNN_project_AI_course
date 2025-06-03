# pybullet_sim.py

import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2  # 用于图像格式转换
from pathlib import Path  # 用于路径操作


# --- 全局日志记录 (可选，但推荐) ---
# from utils.general import LOGGER # 如果你想使用YOLOv5的LOGGER
# 如果不使用YOLOv5的LOGGER，可以用Python内置的logging或简单的print
def log_info(message):
    print(f"INFO (pybullet_sim): {message}")


def log_warning(message):
    print(f"WARNING (pybullet_sim): {message}")


def log_error(message):
    print(f"ERROR (pybullet_sim): {message}")


# ------------------------------------

class SimpleSimulation:
    def __init__(self, connection_mode=p.GUI, urdf_root_path=None):
        """
        初始化PyBullet仿真环境。

        Args:
            connection_mode (int): PyBullet连接模式 (p.GUI 或 p.DIRECT)。
            urdf_root_path (str, optional): 自定义URDF模型的根路径。如果为None，则主要依赖pybullet_data。
        """
        log_info(f"Attempting to connect to PyBullet in {'GUI' if connection_mode == p.GUI else 'DIRECT'} mode.")
        self.physics_client = -1
        self.connection_mode = connection_mode  # <--- 保存连接模式

        try:
            self.physics_client = p.connect(self.connection_mode)
        except p.error as e:
            log_warning(f"PyBullet GUI connection failed (Error: {e}). This can happen on headless servers "
                        "or if X11 forwarding is not set up. Trying p.DIRECT mode.")
            try:
                self.connection_mode = p.DIRECT  # <--- 如果GUI失败，强制切换到DIRECT
                self.physics_client = p.connect(self.connection_mode)
                log_info("Successfully connected to PyBullet in DIRECT mode (no GUI).")
            except Exception as e_direct:
                log_error(f"Failed to connect to PyBullet in DIRECT mode as well: {e_direct}")
                raise

        # 添加 PyBullet 自带的数据路径
        try:
            search_path = pybullet_data.getDataPath()
            p.setAdditionalSearchPath(search_path, physicsClientId=self.physics_client)
            log_info(f"Added PyBullet data search path: {search_path}")
        except Exception as e:
            log_error(f"Could not set pybullet_data search path: {e}")

        if urdf_root_path:
            custom_path = str(Path(urdf_root_path).resolve())
            p.setAdditionalSearchPath(custom_path, physicsClientId=self.physics_client)
            log_info(f"Added custom URDF search path: {custom_path}")

        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

        try:
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
            log_info(f"Loaded plane.urdf (ID: {self.plane_id}).")
        except p.error as e:
            log_error(f"Failed to load plane.urdf: {e}. Simulation might not work correctly.")
            self.plane_id = -1

        self.robot_id = -1
        self.robot_type = None
        self._load_robot()
        self._add_obstacles()
        self._setup_camera()

    def _load_robot(self):
        """加载机器人模型，优先加载racecar，失败则加载cube作为备用。"""
        try:
            self.robot_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.1], physicsClientId=self.physics_client)
            self.robot_type = 'racecar'
            log_info(f"Loaded 'racecar/racecar.urdf' (ID: {self.robot_id}).")
            self._get_racecar_joints()
        except p.error as e_racecar:
            log_warning(f"Failed to load 'racecar/racecar.urdf': {e_racecar}. Trying to load cube.urdf as fallback.")
            try:
                self.robot_id = p.loadURDF("cube.urdf", [0, 0, 0.5], globalScaling=0.3,
                                           physicsClientId=self.physics_client)
                self.robot_type = 'cube'
                log_info(f"Loaded 'cube.urdf' as fallback robot (ID: {self.robot_id}).")
                self.steering_joints = []  # cube没有这些
                self.motorized_joints = []
            except p.error as e_cube:
                log_error(f"Failed to load fallback 'cube.urdf': {e_cube}. No robot loaded.")
                self.robot_id = -1

    def _get_racecar_joints(self):
        """动态获取racecar的转向和驱动轮关节索引。"""
        self.steering_joints = []
        self.motorized_joints = []

        if self.robot_id < 0 or self.robot_type != 'racecar':
            return

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        log_info(f"Racecar has {num_joints} joints. Analyzing...")

        # 标准 racecar.urdf (来自 pybullet_data/racecar) 的关节通常如下：
        # 0: front_left_steering_joint (控制左前轮转向)
        # 1: front_left_wheel_joint (左前轮，可驱动)
        # 2: front_right_steering_joint (控制右前轮转向)
        # 3: front_right_wheel_joint (右前轮，可驱动)
        # 4: rear_left_wheel_joint (左后轮，通常是主要驱动轮)
        # 5: rear_right_wheel_joint (右后轮，通常是主要驱动轮)
        # 后面还有一些 chassis_bottom_joint 等固定关节

        # 简单的基于名称的查找 (可能需要根据你的具体racecar.urdf调整关键词)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = info[1].decode('utf-8').lower()
            joint_type = info[2]

            if 'steering' in joint_name:
                self.steering_joints.append(i)
            # 假设驱动轮是那些名为 "wheel" 且非 "steering" 的旋转关节
            elif 'wheel' in joint_name and 'steering' not in joint_name and \
                    (joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_CONTINUOUS):
                self.motorized_joints.append(i)

        # 如果自动查找效果不好，可以硬编码已知模型的索引
        if not self.steering_joints or not self.motorized_joints:
            log_warning("Automatic joint detection for racecar might have failed. Using default indices.")
            # 适用于 pybullet_data/racecar/racecar.urdf 的常见索引
            self.steering_joints = [0, 2]  # 前轮转向关节
            self.motorized_joints = [4, 5]  # 后轮驱动关节 (也可以用 [1,3,4,5] 驱动所有轮子)
            # 或者只驱动前轮： self.motorized_joints = [1, 3]

        log_info(
            f"Racecar identified/defaulted joints: Steering={self.steering_joints}, Motorized={self.motorized_joints}")

    def _add_obstacles(self):
        """在场景中添加一些静态障碍物。"""
        try:
            p.loadURDF("cube.urdf", [2, 0.5, 0.25], useFixedBase=True, globalScaling=0.5,
                       physicsClientId=self.physics_client)
            p.loadURDF("cube.urdf", [1.5, -0.8, 0.15], useFixedBase=True, globalScaling=0.3,
                       physicsClientId=self.physics_client)
            log_info("Added some cube obstacles.")
        except p.error as e:
            log_warning(f"Could not load default obstacle cubes: {e}")

    def _setup_camera(self):
        """设置虚拟相机的参数。"""
        self.cam_target_pos = [1.5, 0, 0.1]  # 相机看向的目标点 (世界坐标系, 调整到赛车前方)
        self.cam_distance = 2.8  # 相机与目标点的距离
        self.cam_yaw = 70  # 相机水平旋转角度 (让相机稍微偏向赛车路径)
        self.cam_pitch = -25  # 相机俯仰角度 (向下看赛车和前方)
        self.cam_roll = 0
        self.cam_up_axis_index = 2

        self.cam_fov = 65
        self.cam_aspect_ratio = 1.0  # 调整为你获取图像的 width/height
        self.cam_near_plane = 0.01
        self.cam_far_plane = 20

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_target_pos,
            distance=self.cam_distance,
            yaw=self.cam_yaw,
            pitch=self.cam_pitch,
            roll=self.cam_roll,
            upAxisIndex=self.cam_up_axis_index,
            physicsClientId=self.physics_client
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.cam_fov,
            aspect=self.cam_aspect_ratio,
            nearVal=self.cam_near_plane,
            farVal=self.cam_far_plane,
            physicsClientId=self.physics_client
        )
        log_info("Virtual camera set up.")

    def get_camera_image(self, width=320, height=240):
        """获取当前虚拟相机的图像帧。"""
        if self.physics_client < 0:
            log_error("PyBullet not connected. Cannot get camera image.")
            return np.zeros((height, width, 3), dtype=np.uint8)  # 返回黑色图像

        renderer_to_use = p.ER_BULLET_HARDWARE_OPENGL
        # 使用初始化时保存的 self.connection_mode 来判断
        if self.connection_mode == p.DIRECT:  # <--- 修改了这里的判断
            renderer_to_use = p.ER_TINY_RENDERER
            # log_info("Using ER_TINY_RENDERER for DIRECT mode camera.")

        try:
            # 更新投影矩阵的宽高比以匹配请求的图像尺寸
            current_aspect_ratio = width / height
            if abs(current_aspect_ratio - self.cam_aspect_ratio) > 1e-3:  # 如果宽高比变了，重新计算投影矩阵
                self.cam_aspect_ratio = current_aspect_ratio
                self.projection_matrix = p.computeProjectionMatrixFOV(
                    fov=self.cam_fov,
                    aspect=self.cam_aspect_ratio,  # 使用当前的宽高比
                    nearVal=self.cam_near_plane,
                    farVal=self.cam_far_plane,
                    physicsClientId=self.physics_client
                )

            img_arr = p.getCameraImage(
                width,
                height,
                self.view_matrix,
                self.projection_matrix,
                renderer=renderer_to_use,
                physicsClientId=self.physics_client
            )
            w_img, h_img, rgba_px, _, _ = img_arr
            rgba_array = np.array(rgba_px, dtype=np.uint8).reshape((h_img, w_img, 4))
            bgr_frame = cv2.cvtColor(rgba_array[:, :, :3], cv2.COLOR_RGB2BGR)
            return bgr_frame
        except Exception as e:
            log_error(f"Error in get_camera_image: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

    def execute_action(self, action_command_str):
        """根据输入的动作指令字符串，控制仿真机器人执行动作。"""
        if self.robot_id < 0:
            # log_warning("Robot not loaded, cannot execute action.") # 避免过于频繁的日志
            return

        target_velocity = 0.0
        steering_angle = 0.0

        if action_command_str == "MOVE_FORWARD":
            target_velocity = 5.0  # 赛车速度可以快一点
        elif action_command_str == "STOP":
            target_velocity = 0.0
            steering_angle = 0.0  # 停车时方向盘回正
        elif action_command_str == "TURN_LEFT_SLIGHTLY":
            target_velocity = 2.5  # 转弯时减速
            steering_angle = 0.35  # 转向角度，单位是弧度
        elif action_command_str == "TURN_RIGHT_SLIGHTLY":
            target_velocity = 2.5
            steering_angle = -0.35
        elif action_command_str == "BACK_UP_SLIGHTLY":
            target_velocity = -3.0
        else:
            pass  # 未知命令或保持当前状态

        if self.robot_type == 'racecar':
            if self.steering_joints and self.motorized_joints:
                for joint_index in self.steering_joints:
                    p.setJointMotorControl2(self.robot_id, joint_index, p.POSITION_CONTROL,
                                            targetPosition=steering_angle, physicsClientId=self.physics_client)
                for joint_index in self.motorized_joints:
                    p.setJointMotorControl2(self.robot_id, joint_index, p.VELOCITY_CONTROL,
                                            targetVelocity=target_velocity, force=20,
                                            physicsClientId=self.physics_client)  # 增加一点力
            # else: # 已在_get_racecar_joints中打印警告
            # log_warning("Racecar joints not properly identified, cannot control.")
        elif self.robot_type == 'cube':
            linear_vel = [0, 0, 0]
            if action_command_str == "MOVE_FORWARD":
                linear_vel = [0.5, 0, 0]  # cube移动慢一点
            elif action_command_str == "STOP":
                linear_vel = [0, 0, 0]
            elif action_command_str == "BACK_UP_SLIGHTLY":
                linear_vel = [-0.25, 0, 0]
            p.resetBaseVelocity(self.robot_id, linearVelocity=linear_vel, angularVelocity=[0, 0, 0],
                                physicsClientId=self.physics_client)
        # else:
        # log_warning(f"Robot type '{self.robot_type}' not recognized for action '{action_command_str}'")

    def step(self):
        """推进物理仿真一步。"""
        if self.physics_client >= 0:
            p.stepSimulation(physicsClientId=self.physics_client)

    def close(self):
        """断开与PyBullet物理引擎的连接。"""
        if self.physics_client >= 0:
            try:
                log_info("Disconnecting from PyBullet.")
                p.disconnect(physicsClientId=self.physics_client)
            except p.error as e:
                log_warning(f"Error during PyBullet disconnect (might be already disconnected): {e}")
            finally:
                self.physics_client = -1
        else:
            log_info("PyBullet already disconnected or was not connected.")


# --- (可选) 用于独立测试此模块的示例代码 ---
if __name__ == '__main__':
    log_info("--- Testing pybullet_sim.py ---")

    sim_gui = None
    try:
        log_info("Initializing simulation in GUI mode...")
        sim_gui = SimpleSimulation(connection_mode=p.GUI)
        log_info("GUI Simulation initialized.")

        # 设置一个固定的相机视角，方便观察
        # p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=70, cameraPitch=-25,
        #                            cameraTargetPosition=[1.5,0,0.1], physicsClientId=sim_gui.physics_client)

        for i in range(720):  # 运行大约12秒 (假设240Hz物理模拟，演示循环控制在30-60fps)
            action = "STOP"  # 默认动作
            if 0 <= i < 120:  # 前2秒前进
                action = "MOVE_FORWARD"
            elif 120 <= i < 240:  # 左转2秒
                action = "TURN_LEFT_SLIGHTLY"
            elif 240 <= i < 360:  # 前进2秒
                action = "MOVE_FORWARD"
            elif 360 <= i < 480:  # 右转2秒
                action = "TURN_RIGHT_SLIGHTLY"
            elif 480 <= i < 600:  # 后退2秒
                action = "BACK_UP_SLIGHTLY"
            else:  # 最后停止
                action = "STOP"

            sim_gui.execute_action(action)
            sim_gui.step()

            if i % 8 == 0:  # 大约 30 FPS (240Hz / 8) 获取图像
                frame = sim_gui.get_camera_image(width=640, height=480)
                if frame is not None and frame.size > 0:
                    # log_info(f"Frame {i//8}: Got camera image of shape {frame.shape}")
                    cv2.imshow("PyBullet Sim Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log_info("'q' pressed, exiting test loop.")
                        break
                else:
                    log_warning(f"Frame {i // 8}: Failed to get camera image.")

            time.sleep(1. / 240.)  # 模拟物理引擎的典型步长时间

    except Exception as e:
        log_error(f"Error during GUI simulation test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if sim_gui:
            sim_gui.close()
        cv2.destroyAllWindows()
        log_info("GUI simulation test finished.")