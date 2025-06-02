# Pybullet_control.py

#我最终设置的是在仿真环境下构建一个双轮机器人（双轮小车）+相机，每隔3秒给相机传一张现实照片，模拟现实，以此观察小车的运动

#准备：
# 一、照片数据集：
# 创建一个名为 real_world_photos 的文件夹
# 放入至少 5-10 张现实世界的 JPG/PNG 格式照片
# 照片内容应包含可能遇到的障碍物（如各种家具等）

#二、下载pip install pybullet numpy opencv-python torch torchvision glob2



import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import time
import os
import glob

from decision_maker import get_action_command, ACTION_STOP, ACTION_MOVE_FORWARD, ACTION_TURN_LEFT_SLIGHTLY, \
    ACTION_TURN_RIGHT_SLIGHTLY


def load_photo_sequence(photo_folder):
    """加载照片序列并按文件名排序"""
    photo_paths = glob.glob(os.path.join(photo_folder, "*.jpg")) + \
                  glob.glob(os.path.join(photo_folder, "*.png")) + \
                  glob.glob(os.path.join(photo_folder, "*.jpeg"))
    photo_paths.sort()  # 按文件名排序
    print(f"Loaded {len(photo_paths)} photos from {photo_folder}")
    return photo_paths

class PhotoSequenceSimulator:
    """模拟现实世界照片序列输入"""

    def __init__(self, photo_paths, time_per_photo=3.0, loop=True):
        self.photo_paths = photo_paths
        self.time_per_photo = time_per_photo
        self.loop = loop
        self.current_index = 0
        self.last_switch_time = time.time()
        self.current_photo = None
        self.load_current_photo()

    def load_current_photo(self):
        """加载当前照片"""
        if not self.photo_paths:
            return None

        path = self.photo_paths[self.current_index]
        self.current_photo = cv2.imread(path)
        if self.current_photo is None:
            print(f"Error loading photo: {path}")
            return None

        print(f"Displaying photo: {os.path.basename(path)}")
        return self.current_photo

    def update(self):
        """更新照片序列"""
        if not self.photo_paths or len(self.photo_paths) <= 1:
            return self.current_photo

        current_time = time.time()
        if current_time - self.last_switch_time > self.time_per_photo:
            self.last_switch_time = current_time
            self.current_index += 1

            if self.current_index >= len(self.photo_paths):
                if self.loop:
                    self.current_index = 0
                else:
                    self.current_index = len(self.photo_paths) - 1

            self.load_current_photo()

        return self.current_photo


# 初始化PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 创建仿真环境
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 0.1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

# 创建双轮机器人
robot = p.loadURDF("r2d2.urdf", startPos, startOrientation)
num_joints = p.getNumJoints(robot)
wheel_joints = [2, 3]  # R2D2模型的轮子关节索引


# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # 置信度阈值

PHOTO_FOLDER = "real_world_photos"  # 存放现实照片的文件夹
photo_paths = load_photo_sequence(PHOTO_FOLDER)
photo_simulator = PhotoSequenceSimulator(photo_paths, time_per_photo=5.0, loop=True)

# 相机参数设置
camera_width, camera_height = 640, 480

# 机器人控制参数
BASE_SPEED = 5.0
TURN_FACTOR = 0.5
BACKUP_DURATION = 0.5  # 后退持续时间(秒)
backup_timer = 0

# 创建OpenCV窗口
cv2.namedWindow("Real World View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real World View", camera_width, camera_height)

# 主仿真循环
try:
    while True:
        real_world_img = photo_simulator.update()

        if real_world_img is None:
            print("Error: No valid photos available. Using blank image.")
            real_world_img = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)

        # 调整照片尺寸以匹配预期分辨率
        real_world_img = cv2.resize(real_world_img, (camera_width, camera_height))

        # 使用YOLOv5进行障碍物检测
        results = model(real_world_img)
        detections = results.pandas().xyxy[0]

        # 格式化检测结果
        detections_list = []
        for _, detection in detections.iterrows():
            detections_list.append({
                'class_id': detection['class'],
                'class_name': detection['name'],
                'bbox': [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']],
                'confidence': detection['confidence']
            })

        # 调用决策模块
        action = get_action_command(detections_list, camera_width, camera_height)
        print(f"Action: {action}")

        # 在图像上绘制检测结果
        processed_img = real_world_img.copy()
        for det in detections_list:
            xmin, ymin, xmax, ymax = map(int, det['bbox'])
            cv2.rectangle(processed_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(processed_img, f"{det['class_name']} {det['confidence']:.2f}",
                        (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow("Real World View", processed_img)

        # 获取机器人位置和方向（仅用于仿真可视化）
        robot_pos, robot_orn = p.getBasePositionAndOrientation(robot)

        # 执行动作控制
        left_wheel_speed = 0
        right_wheel_speed = 0

        if backup_timer > 0:
            # 正在后退
            left_wheel_speed = -BASE_SPEED
            right_wheel_speed = -BASE_SPEED
            backup_timer -= 1 / 240.0
        else:
            if action == ACTION_MOVE_FORWARD:
                left_wheel_speed = BASE_SPEED
                right_wheel_speed = BASE_SPEED
            elif action == ACTION_TURN_LEFT_SLIGHTLY:
                left_wheel_speed = BASE_SPEED * (1 - TURN_FACTOR)
                right_wheel_speed = BASE_SPEED * (1 + TURN_FACTOR)
            elif action == ACTION_TURN_RIGHT_SLIGHTLY:
                left_wheel_speed = BASE_SPEED * (1 + TURN_FACTOR)
                right_wheel_speed = BASE_SPEED * (1 - TURN_FACTOR)
            elif action == ACTION_STOP:
                # 遇到障碍物时先停止，然后后退
                backup_timer = BACKUP_DURATION

        # 应用轮子速度（仅影响仿真机器人）
        p.setJointMotorControlArray(
            robot,
            wheel_joints,
            p.VELOCITY_CONTROL,
            targetVelocities=[left_wheel_speed, right_wheel_speed]
        )

        # 步进仿真
        p.stepSimulation()

        # 处理按键事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):  # 手动切换到下一张照片
            photo_simulator.current_index = (photo_simulator.current_index + 1) % len(photo_simulator.photo_paths)
            photo_simulator.load_current_photo()
        elif key == ord('p'):  # 手动切换到上一张照片
            photo_simulator.current_index = (photo_simulator.current_index - 1) % len(photo_simulator.photo_paths)
            photo_simulator.load_current_photo()

except Exception as e:
    print(f"Error: {e}")

finally:
    p.disconnect()
    cv2.destroyAllWindows()