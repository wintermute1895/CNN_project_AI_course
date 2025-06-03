# decision_maker.py (极简版，你之前提供的版本稍作调整)

# --- 定义动作指令常量 ---
ACTION_STOP = "STOP"
ACTION_MOVE_FORWARD = "MOVE_FORWARD"
ACTION_TURN_LEFT = "TURN_LEFT_SLIGHTLY"
ACTION_TURN_RIGHT = "TURN_RIGHT_SLIGHTLY"
ACTION_BACK_UP = "BACK_UP_SLIGHTLY"
# ------------------------

# --- 决策参数 (可以根据需要调整或从外部配置加载) ---
# TODO: 【可调参数】根据你的场景和机器人调整这些阈值
DANGER_ZONE_X_CENTER_RATIO = 0.5  # 危险区中心线在图像X轴的比例
DANGER_ZONE_WIDTH_RATIO = 0.4     # 危险区宽度占图像宽度的比例
DANGER_ZONE_Y_BOTTOM_RATIO = 0.8  # 危险区底部在图像Y轴的比例 (从顶部0到底部1)
DANGER_ZONE_Y_TOP_RATIO = 0.3     # 危险区顶部在图像Y轴的比例 (更关注图像中下部)

OBSTACLE_CONFIDENCE_THRESHOLD = 0.4 # TODO: 【可调参数】认为检测有效的最低置信度
LARGE_OBSTACLE_AREA_RATIO = 0.05   # TODO: 【可调参数】障碍物占图像面积超过此比例视为大障碍物
# -------------------------------------------------

def get_bbox_center_and_area(bbox):
    xmin, ymin, xmax, ymax = bbox
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    area = (xmax - xmin) * (ymax - ymin)
    return center_x, center_y, area

def get_action_command_simple(detections_list, image_width, image_height):
    if not detections_list: # 如果没有检测到物体
        # print("Decision: No obstacles, MOVE_FORWARD")
        return ACTION_MOVE_FORWARD

    # 定义危险区域的边界
    danger_zone_xmin = image_width * (DANGER_ZONE_X_CENTER_RATIO - DANGER_ZONE_WIDTH_RATIO / 2)
    danger_zone_xmax = image_width * (DANGER_ZONE_X_CENTER_RATIO + DANGER_ZONE_WIDTH_RATIO / 2)
    danger_zone_ymin = image_height * DANGER_ZONE_Y_TOP_RATIO
    danger_zone_ymax = image_height * DANGER_ZONE_Y_BOTTOM_RATIO

    # 寻找在危险区域内且置信度最高的“最危险”障碍物
    most_dangerous_obstacle = None
    max_confidence_in_danger_zone = 0

    for det in detections_list:
        if det['confidence'] < OBSTACLE_CONFIDENCE_THRESHOLD:
            continue

        bbox = det['bbox']
        obj_center_x, obj_center_y, obj_area = get_bbox_center_and_area(bbox)

        # 判断物体中心是否在危险区域内
        is_in_danger = (danger_zone_xmin < obj_center_x < danger_zone_xmax and
                        danger_zone_ymin < obj_center_y < danger_zone_ymax)

        if is_in_danger:
            # print(f"Debug: Obstacle '{det['class_name']}' in danger zone. Center: ({obj_center_x:.0f}, {obj_center_y:.0f})")
            if det['confidence'] > max_confidence_in_danger_zone:
                max_confidence_in_danger_zone = det['confidence']
                most_dangerous_obstacle = {
                    'center_x': obj_center_x,
                    'area_ratio': obj_area / (image_width * image_height),
                    'class_name': det['class_name'] # 可以根据类别增加不同权重
                }

    if most_dangerous_obstacle:
        # 如果大障碍物在正前方，优先停止或后退
        if most_dangerous_obstacle['area_ratio'] > LARGE_OBSTACLE_AREA_RATIO:
            # print(f"Decision: Large obstacle ({most_dangerous_obstacle['class_name']}) in danger zone, STOP.")
            return ACTION_STOP # 或者 ACTION_BACK_UP

        # 根据最危险障碍物的位置决定转向
        if most_dangerous_obstacle['center_x'] < image_width * DANGER_ZONE_X_CENTER_RATIO:
            # print(f"Decision: Obstacle ({most_dangerous_obstacle['class_name']}) on left, TURN_RIGHT.")
            return ACTION_TURN_RIGHT
        else:
            # print(f"Decision: Obstacle ({most_dangerous_obstacle['class_name']}) on right, TURN_LEFT.")
            return ACTION_TURN_LEFT
    else: # 危险区域内没有置信度足够的障碍物
        # print("Decision: Danger zone clear, MOVE_FORWARD.")
        return ACTION_MOVE_FORWARD