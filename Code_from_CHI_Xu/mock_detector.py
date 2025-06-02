# mock_detector.py
# 作用: 模拟障碍物检测模块的输出，为决策逻辑和仿真提供测试数据。
# 负责人: A (迟旭) - 用于辅助自己和E同学的开发，直到D同学的真实检测模块可用。

import random  # 用于一些随机性（如果需要的话）

# --- 与 decision_maker.py 约定的动作指令常量 ---
# (这里列出来方便参考，实际使用时 decision_maker.py 会定义它们)
# ACTION_STOP = "STOP"
# ACTION_MOVE_FORWARD = "MOVE_FORWARD"
# ACTION_TURN_LEFT_SLIGHTLY = "TURN_LEFT_SLIGHTLY"
# ACTION_TURN_RIGHT_SLIGHTLY = "TURN_RIGHT_SLIGHTLY"
# ACTION_BACK_UP_SLIGHTLY = "BACK_UP_SLIGHTLY"

# --- 模拟的图像尺寸 ---
# 这个尺寸应该与PyBullet虚拟相机输出的图像尺寸，或D同学检测脚本处理的图像尺寸保持一致。
# 你可以根据需要调整。
MOCK_IMAGE_WIDTH = 640
MOCK_IMAGE_HEIGHT = 480

# --- 预定义的障碍物类别 (与你们项目的 your_dataset.yaml 一致) ---
# 这是一个示例，请替换为你们实际的类别
# 假设你们的类别如下：
# 0: 'person'
# 1: 'car'
# 2: 'cone'
# 3: 'barrier'
# ...
# 21: 'pothole'
MOCK_CLASS_NAMES = {
    0: 'person',
    1: 'car',
    2: 'cone',
    3: 'barrier',
    # ... 添加你们所有的类别直到 21 ...
    21: 'pothole'
}


def _create_detection_object(class_id, bbox_coords, confidence=0.85):
    """辅助函数，创建一个符合接口要求的检测对象字典。"""
    if class_id not in MOCK_CLASS_NAMES:
        raise ValueError(f"未定义的 class_id: {class_id}。请在 MOCK_CLASS_NAMES 中添加。")
    return {
        'class_id': class_id,
        'class_name': MOCK_CLASS_NAMES[class_id],
        'bbox': [float(coord) for coord in bbox_coords],  # 确保是浮点数
        'confidence': float(confidence)
    }


def get_simulated_detections(scenario_id, image_width=MOCK_IMAGE_WIDTH, image_height=MOCK_IMAGE_HEIGHT):
    """
    根据场景ID返回模拟的障碍物检测结果。

    Args:
        scenario_id (int): 场景标识符。
        image_width (int, optional): 模拟图像的宽度。默认为 MOCK_IMAGE_WIDTH。
        image_height (int, optional): 模拟图像的高度。默认为 MOCK_IMAGE_HEIGHT。

    Returns:
        tuple: (detections_list, image_height, image_width)
            detections_list (list): 模拟的检测结果列表。
            image_height (int): 图像高度。
            image_width (int): 图像宽度。
    """
    detections = []

    # --- 根据场景ID生成不同的模拟数据 ---

    if scenario_id == 0:
        # 场景0: 没有障碍物
        pass  # detections 保持为空列表

    elif scenario_id == 1:
        # 场景1: 一个障碍物在图像正中央的危险区域
        # 假设类别ID为 2 ('cone')
        center_x = image_width * 0.5
        center_y = image_height * 0.6  # 稍微偏下一点
        obj_width = image_width * 0.1
        obj_height = image_height * 0.15

        xmin = center_x - obj_width / 2
        ymin = center_y - obj_height / 2
        xmax = center_x + obj_width / 2
        ymax = center_y + obj_height / 2
        detections.append(_create_detection_object(2, [xmin, ymin, xmax, ymax], 0.9))

    elif scenario_id == 2:
        # 场景2: 一个障碍物在危险区域的左侧
        # 假设类别ID为 3 ('barrier')
        center_x = image_width * 0.3  # 偏左
        center_y = image_height * 0.7
        obj_width = image_width * 0.15
        obj_height = image_height * 0.1

        xmin = center_x - obj_width / 2
        ymin = center_y - obj_height / 2
        xmax = center_x + obj_width / 2
        ymax = center_y + obj_height / 2
        detections.append(_create_detection_object(3, [xmin, ymin, xmax, ymax], 0.8))

    elif scenario_id == 3:
        # 场景3: 一个障碍物在危险区域的右侧
        # 假设类别ID为 1 ('car')
        center_x = image_width * 0.7  # 偏右
        center_y = image_height * 0.55
        obj_width = image_width * 0.2
        obj_height = image_height * 0.2

        xmin = center_x - obj_width / 2
        ymin = center_y - obj_height / 2
        xmax = center_x + obj_width / 2
        ymax = center_y + obj_height / 2
        detections.append(_create_detection_object(1, [xmin, ymin, xmax, ymax], 0.92))

    elif scenario_id == 4:
        # 场景4: 多个障碍物，一个在左，一个在右，都在危险区
        # 左侧障碍物 (cone)
        detections.append(_create_detection_object(2,
                                                   [image_width * 0.2, image_height * 0.6, image_width * 0.3,
                                                    image_height * 0.7], 0.85))
        # 右侧障碍物 (barrier)
        detections.append(_create_detection_object(3,
                                                   [image_width * 0.7, image_height * 0.5, image_width * 0.8,
                                                    image_height * 0.6], 0.88))

    elif scenario_id == 5:
        # 场景5: 一个大的障碍物在正前方，非常近
        # 假设类别ID为 21 ('pothole')，但表现为一个大障碍
        center_x = image_width * 0.5
        center_y = image_height * 0.75  # 更靠近底部
        obj_width = image_width * 0.5  # 占据一半宽度
        obj_height = image_height * 0.3  # 占据30%高度

        xmin = center_x - obj_width / 2
        ymin = center_y - obj_height / 2
        xmax = center_x + obj_width / 2
        ymax = center_y + obj_height / 2  # ymax会比较大，表示近
        detections.append(_create_detection_object(21, [xmin, ymin, xmax, ymax], 0.95))

    elif scenario_id == 6:
        # 场景6: 多个障碍物，一个在危险区，一个在危险区外
        # 危险区内 (cone)
        detections.append(_create_detection_object(2,
                                                   [image_width * 0.45, image_height * 0.55, image_width * 0.55,
                                                    image_height * 0.65], 0.9))
        # 危险区外 (car, 远处左上角)
        detections.append(_create_detection_object(1, [10, 10, 80, 80], 0.7))

    # 你可以根据需要添加更多的场景 (scenario_id)

    else:
        print(f"WARNING: mock_detector: 未知的 scenario_id: {scenario_id}. 返回空检测结果。")
        # 默认返回空列表

    return detections, image_height, image_width


# --- 主测试块：用于独立测试这个模拟器脚本 ---
if __name__ == "__main__":
    print("--- Testing Mock Detector Script ---")

    scenarios_to_test = [0, 1, 2, 3, 4, 5, 6, 99(不存在的场景)]

    for sid in scenarios_to_test:
        print(f"\n--- Testing Scenario ID: {sid} ---")
        simulated_detections, height, width = get_simulated_detections(sid)

        print(f"Image Size: {width}x{height}")
        if simulated_detections:
            print(f"Detected {len(simulated_detections)} objects:")
            for i, det in enumerate(simulated_detections):
                print(f"  Object {i + 1}:")
                print(f"    Class ID: {det['class_id']}")
                print(f"    Class Name: {det['class_name']}")
                print(f"    BBox (xmin,ymin,xmax,ymax): {det['bbox']}")
                print(f"    Confidence: {det['confidence']:.2f}")
        else:
            print("No objects detected in this scenario.")

    print("\n--- Mock Detector Test Finished ---")
    