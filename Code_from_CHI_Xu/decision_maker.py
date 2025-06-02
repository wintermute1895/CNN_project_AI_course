# decision_maker.py

# -----------------------------------------------------------------------------
# 模块：决策逻辑模块 (Decision Maker)
# 目标：根据检测到的障碍物信息，输出一个简单的动作指令用于仿真控制。
# -----------------------------------------------------------------------------

# === 1. 定义动作指令常量 ===
# 这些是我们的仿真机器人可以理解和执行的抽象动作。
# 后续 E 同学在 PyBullet 控制脚本中，会根据这些指令字符串调用具体的机器人动作函数。
ACTION_STOP = "STOP"  # 机器人停止
ACTION_MOVE_FORWARD = "MOVE_FORWARD"  # 机器人向前直线移动
ACTION_TURN_LEFT_SLIGHTLY = "TURN_LEFT_SLIGHTLY"  # 机器人轻微向左转弯
ACTION_TURN_RIGHT_SLIGHTLY = "TURN_RIGHT_SLIGHTLY"  # 机器人轻微向右转弯
ACTION_BACK_UP_SLIGHTLY = "BACK_UP_SLIGHTLY"  # 机器人轻微后退 (新增)
# 你可以根据需要添加更多动作指令，例如 ACTION_TURN_LEFT_SHARPLY 等。

# === 2. 配置参数 ===
# 这些参数可以根据实际测试效果进行调整，甚至可以考虑放到一个单独的配置文件中。

# --- 危险区域定义 (相对于图像尺寸的比例) ---
# 假设机器人的前进方向是图像的正前方中心区域。
DANGER_ZONE_CENTER_X_RATIO = 0.5  # 危险区域的中心线在图像宽度的50%位置
DANGER_ZONE_WIDTH_RATIO = 0.6  # 危险区域的宽度占图像总宽度的60% (即从20%到80%区域)
DANGER_ZONE_MIN_Y_RATIO = 0.4  # 危险区域从图像高度的40%开始 (从顶部算起)
DANGER_ZONE_MAX_Y_RATIO = 0.9  # 危险区域到图像高度的90%结束 (从顶部算起, 底部留一些空间)

# --- 障碍物属性阈值 ---
CRITICAL_OBSTACLE_CONFIDENCE = 0.5  # 认为检测结果有效的最低置信度
# 障碍物面积占图像总面积的比例，超过此值认为是“大”障碍物，可能需要更谨慎的响应
LARGE_OBSTACLE_AREA_RATIO_THRESHOLD = 0.15
# “非常近”的障碍物的Y坐标上限比例 (ymax 越接近1，说明越靠近图像底部，即离机器人越近)
VERY_CLOSE_OBSTACLE_YMAX_RATIO_THRESHOLD = 0.85


# --- (可选) 特定类别的威胁等级或特殊处理 ---
# 例如，你可以定义一个列表或字典来标记哪些障碍物类别需要特别注意
# CRITICAL_CLASS_NAMES = ["person", "large_rock", "vehicle"] # 示例

# === 3. 辅助函数 ===

def get_bbox_center(bbox):
    """计算边界框的中心点坐标 (cx, cy)。"""
    xmin, ymin, xmax, ymax = bbox
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def get_bbox_area(bbox):
    """计算边界框的面积。"""
    xmin, ymin, xmax, ymax = bbox
    return (xmax - xmin) * (ymax - ymin)


def is_in_danger_zone(bbox_center_x, bbox_center_y, bbox_ymax, image_width, image_height):
    """
    判断障碍物的中心点是否落入预定义的危险区域。
    同时考虑障碍物的ymax是否非常靠近底部，表示很近。
    """
    danger_zone_xmin = image_width * (DANGER_ZONE_CENTER_X_RATIO - DANGER_ZONE_WIDTH_RATIO / 2)
    danger_zone_xmax = image_width * (DANGER_ZONE_CENTER_X_RATIO + DANGER_ZONE_WIDTH_RATIO / 2)
    danger_zone_ymin = image_height * DANGER_ZONE_MIN_Y_RATIO
    danger_zone_ymax_boundary = image_height * DANGER_ZONE_MAX_Y_RATIO  # 危险区域的远端边界

    in_horizontal_range = danger_zone_xmin < bbox_center_x < danger_zone_xmax
    in_vertical_range = danger_zone_ymin < bbox_center_y < danger_zone_ymax_boundary

    is_very_close = bbox_ymax > (image_height * VERY_CLOSE_OBSTACLE_YMAX_RATIO_THRESHOLD)

    if in_horizontal_range and in_vertical_range:
        return True, is_very_close  # 返回是否在危险区，以及是否非常近
    return False, False


# === 4. 核心决策函数 ===

def get_action_command(detections_list, image_width, image_height):
    """
    根据检测到的障碍物列表和图像尺寸，决定机器人应该执行的动作指令。

    Args:
        detections_list (list): 检测到的障碍物列表，每个元素是字典:
            {'class_id': int, 'class_name': str,
             'bbox': [xmin, ymin, xmax, ymax], 'confidence': float}
        image_width (int): 图像宽度。
        image_height (int): 图像高度。

    Returns:
        str: 动作指令字符串。
    """
    # print(f"DEBUG: Decision_maker received {len(detections_list)} detections. Image size: {image_width}x{image_height}") # 调试信息

    if not detections_list:
        # print("INFO: No obstacles detected. Suggesting MOVE_FORWARD.")
        return ACTION_MOVE_FORWARD

    # --- 分析危险区域内的障碍物 ---
    obstacles_in_danger_zone = []
    for det in detections_list:
        if det['confidence'] >= CRITICAL_OBSTACLE_CONFIDENCE:
            bbox = det['bbox']
            bbox_center_x, bbox_center_y = get_bbox_center(bbox)

            in_danger, is_close = is_in_danger_zone(bbox_center_x, bbox_center_y, bbox[3], image_width, image_height)

            if in_danger:
                obstacles_in_danger_zone.append({
                    'center_x': bbox_center_x,
                    'area_ratio': get_bbox_area(bbox) / (image_width * image_height),
                    'is_very_close': is_close,
                    'class_name': det['class_name']
                    # 可以加入更多信息，如原始bbox，用于更复杂的决策
                })

    # print(f"DEBUG: Found {len(obstacles_in_danger_zone)} critical obstacles in danger zone.") # 调试信息

    if not obstacles_in_danger_zone:
        # print("INFO: No critical obstacles in danger zone. Suggesting MOVE_FORWARD.")
        return ACTION_MOVE_FORWARD

    # --- 基于危险区域内障碍物的简单决策规则 ---
    # 优先处理又大又近的障碍物
    for obs in obstacles_in_danger_zone:
        if obs['is_very_close'] and obs['area_ratio'] > LARGE_OBSTACLE_AREA_RATIO_THRESHOLD:
            # print(f"INFO: Large and very close obstacle '{obs['class_name']}' detected. Suggesting STOP or BACK_UP.")
            return ACTION_STOP  # 或者 ACTION_BACK_UP_SLIGHTLY
        if obs['is_very_close']:  # 如果有任何非常近的障碍物
            # print(f"INFO: Very close obstacle '{obs['class_name']}' detected. Suggesting STOP.")
            return ACTION_STOP

    # 如果没有特别近的，但危险区有障碍物，尝试避让
    # 统计左右两侧障碍物的“威胁度”（可以简单地用数量或面积加权）
    left_threat = 0
    right_threat = 0
    center_line_x = image_width * DANGER_ZONE_CENTER_X_RATIO

    for obs in obstacles_in_danger_zone:
        # 简单的威胁度计算：可以考虑面积、距离（通过y坐标估算）等
        # 这里我们简单地看障碍物中心在中心线的哪一侧
        if obs['center_x'] < center_line_x:
            left_threat += 1  # 或者 += obs['area_ratio']
        else:
            right_threat += 1  # 或者 += obs['area_ratio']

    if left_threat > 0 and right_threat == 0:  # 障碍物主要在左侧
        # print(f"INFO: Obstacles primarily on the left in danger zone ({left_threat} vs {right_threat}). Suggesting TURN_RIGHT_SLIGHTLY.")
        return ACTION_TURN_RIGHT_SLIGHTLY
    elif right_threat > 0 and left_threat == 0:  # 障碍物主要在右侧
        # print(f"INFO: Obstacles primarily on the right in danger zone ({left_threat} vs {right_threat}). Suggesting TURN_LEFT_SLIGHTLY.")
        return ACTION_TURN_LEFT_SLIGHTLY
    elif left_threat > 0 and right_threat > 0:  # 两侧都有障碍物，或者障碍物横跨中心
        # print(f"INFO: Obstacles on both sides or centered in danger zone ({left_threat} vs {right_threat}). Suggesting STOP.")
        return ACTION_STOP
    else:  # 逻辑上这个分支不应该被走到，因为上面已经判断过 obstacles_in_danger_zone 非空
        # print("INFO: Fallback, no specific rule matched for obstacles in danger zone. Suggesting STOP.")
        return ACTION_STOP


# === 5. 独立测试该脚本的逻辑 (main 测试块) ===
if __name__ == "__main__":
    print("--- Testing Decision Maker Script ---")

    # 模拟图像尺寸
    img_w, img_h = 640, 480
    print(f"Simulated image size: {img_w}x{img_h}")

    # 测试用例 1: 没有障碍物
    print("\n--- Test Case 1: No Obstacles ---")
    mock_detections_1 = []
    action1 = get_action_command(mock_detections_1, img_w, img_h)
    print(f"Detections: {mock_detections_1}")
    print(f"Action Command: {action1}")
    assert action1 == ACTION_MOVE_FORWARD

    # 测试用例 2: 危险区域中央有一个大障碍物，且很近
    print("\n--- Test Case 2: Large & Close Obstacle in Center Danger Zone ---")
    # 危险区: x from 128 to 512, y from 192 to 432
    # 非常近: ymax > 480 * 0.85 = 408
    mock_detections_2 = [
        {'class_id': 0, 'class_name': 'big_rock',
         'bbox': [img_w * 0.3, img_h * 0.5, img_w * 0.7, img_h * 0.9],
         # 中心(320,336), ymax=432, 面积是图像的0.4*0.4=0.16 > 0.15
         'confidence': 0.9}
    ]
    action2 = get_action_command(mock_detections_2, img_w, img_h)
    print(f"Detections: {mock_detections_2}")
    print(f"Action Command: {action2}")
    assert action2 == ACTION_STOP  # 或 ACTION_BACK_UP_SLIGHTLY

    # 测试用例 3: 危险区域左侧有障碍物
    print("\n--- Test Case 3: Obstacle on Left in Danger Zone ---")
    mock_detections_3 = [
        {'class_id': 1, 'class_name': 'small_box',
         'bbox': [img_w * 0.25, img_h * 0.6, img_w * 0.35, img_h * 0.7],  # 中心(192,312), ymax=336
         'confidence': 0.8}
    ]
    action3 = get_action_command(mock_detections_3, img_w, img_h)
    print(f"Detections: {mock_detections_3}")
    print(f"Action Command: {action3}")
    assert action3 == ACTION_TURN_RIGHT_SLIGHTLY

    # 测试用例 4: 危险区域右侧有障碍物 (但不够近，也不是很大)
    print("\n--- Test Case 4: Obstacle on Right in Danger Zone (not very close/large) ---")
    mock_detections_4 = [
        {'class_id': 1, 'class_name': 'small_cone',
         'bbox': [img_w * 0.65, img_h * 0.5, img_w * 0.75, img_h * 0.6],  # 中心(448,264), ymax=288, 面积0.1*0.1=0.01
         'confidence': 0.8}
    ]
    action4 = get_action_command(mock_detections_4, img_w, img_h)
    print(f"Detections: {mock_detections_4}")
    print(f"Action Command: {action4}")
    assert action4 == ACTION_TURN_LEFT_SLIGHTLY

    # 测试用例 5: 危险区域外有障碍物
    print("\n--- Test Case 5: Obstacle Outside Danger Zone ---")
    mock_detections_5 = [
        {'class_id': 0, 'class_name': 'far_object',
         'bbox': [10, 10, 50, 50],  # 中心(30,30)
         'confidence': 0.9}
    ]
    action5 = get_action_command(mock_detections_5, img_w, img_h)
    print(f"Detections: {mock_detections_5}")
    print(f"Action Command: {action5}")
    assert action5 == ACTION_MOVE_FORWARD

    # 测试用例 6: 低置信度障碍物在危险区
    print("\n--- Test Case 6: Low Confidence Obstacle in Danger Zone ---")
    mock_detections_6 = [
        {'class_id': 0, 'class_name': 'blurry_thing',
         'bbox': [img_w * 0.4, img_h * 0.4, img_w * 0.6, img_h * 0.6],
         'confidence': CRITICAL_OBSTACLE_CONFIDENCE - 0.1}  # 低于阈值
    ]
    action6 = get_action_command(mock_detections_6, img_w, img_h)
    print(f"Detections: {mock_detections_6}")
    print(f"Action Command: {action6}")
    assert action6 == ACTION_MOVE_FORWARD

    # 测试用例 7: 危险区域左右都有障碍物
    print("\n--- Test Case 7: Obstacles on Both Sides in Danger Zone ---")
    mock_detections_7 = [
        {'class_id': 1, 'class_name': 'box_left',
         'bbox': [img_w * 0.25, img_h * 0.6, img_w * 0.35, img_h * 0.7], 'confidence': 0.8},
        {'class_id': 1, 'class_name': 'box_right',
         'bbox': [img_w * 0.65, img_h * 0.5, img_w * 0.75, img_h * 0.6], 'confidence': 0.8}
    ]
    action7 = get_action_command(mock_detections_7, img_w, img_h)
    print(f"Detections: {mock_detections_7}")
    print(f"Action Command: {action7}")
    assert action7 == ACTION_STOP

    print("\n--- All mock tests passed (if no assertion errors) ---")