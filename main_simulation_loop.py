# main_simulation_loop.py

import sys
from pathlib import Path
import os
import time
import cv2  # 用于可选的图像显示
import pybullet as p  # 用于连接类型 p.GUI 或 p.DIRECT

# --- 1. 设置Python模块搜索路径 (sys.path) ---
# 获取当前脚本(main_simulation_loop.py)所在的目录，我们假设这是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 定义YOLOv5源码的根目录路径 (假设为项目根目录下的 'yolov5' 文件夹)
YOLOV5_ROOT = PROJECT_ROOT / "yolov5"

# 将项目根目录和YOLOv5根目录添加到sys.path，确保所有导入都能正确找到
# 项目根目录优先，以便能找到 obstacle_detector.py, pybullet_sim.py, decision_maker.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"DEBUG: Prepended to sys.path: {PROJECT_ROOT}")

if str(YOLOV5_ROOT) not in sys.path:
    # 将YOLOv5的路径插入到第二个位置，这样如果项目根目录有同名模块，项目根目录的优先
    sys.path.insert(1, str(YOLOV5_ROOT))
    print(f"DEBUG: Inserted to sys.path (for YOLOv5 modules): {YOLOV5_ROOT}")

# (可选) 打印 sys.path 确认顺序
# print("DEBUG: sys.path entries:")
# for i, path_entry in enumerate(sys.path):
#     print(f"  [{i}] {path_entry}")
# ---------------------------------------------

# --- 2. 导入你的模块和必要的库 ---
try:
    from obstacle_detector import detect_obstacles_in_image
    from pybullet_sim import SimpleSimulation
    from decision_maker import get_action_command_simple
    # decision_maker 中定义的动作常量也导入一下，方便直接使用
    from decision_maker import ACTION_STOP, ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_BACK_UP

    print("SUCCESS: Custom modules (obstacle_detector, pybullet_sim, decision_maker) imported.")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import one or more custom modules.")
    print(f"  Make sure obstacle_detector.py, pybullet_sim.py, decision_maker.py are in: {PROJECT_ROOT}")
    print(f"  ImportError: {e}")
    sys.exit(1)


# ---------------------------------------------

def main():
    print("--- Starting Embodied AI Simulation Loop ---")

    # --- 3. 初始化仿真环境 ---
    # TODO: 【可配置】选择连接模式: p.GUI (显示图形界面) 或 p.DIRECT (无图形界面，后台运行)
    # 对于本地演示和调试，p.GUI 更好。
    # 如果在没有显示的服务器或Docker中（且未配置X11转发），必须用 p.DIRECT。
    connection_mode = p.GUI
    # connection_mode = p.DIRECT # 如果需要无头运行

    print(f"Initializing PyBullet simulation with mode: {'GUI' if connection_mode == p.GUI else 'DIRECT'}...")
    try:
        sim = SimpleSimulation(connection_mode=connection_mode)
        print("PyBullet simulation initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize PyBullet simulation: {e}")
        print(
            "  Ensure PyBullet is installed correctly and any necessary graphics drivers are available (for GUI mode).")
        return  # 无法继续

    # --- 4. （可选）模型加载信息在 obstacle_detector.py 内部处理 ---
    # obstacle_detector.py 应该在导入时或首次调用时加载模型。
    # 我们可以在这里进行一次虚拟调用以确保模型已加载 (如果检测模块没有在__init__中加载模型)
    print("Verifying obstacle detector (may load model if not already loaded)...")
    try:
        # 用一个全黑的小图像快速测试检测器是否能工作，并获取其期望的图像尺寸信息
        _dummy_frame = cv2.UMat(240, 320, cv2.CV_8UC3).get()  # 创建一个OpenCV UMat然后转为numpy
        _dummy_frame.fill(0)
        _, _, _ = detect_obstacles_in_image(_dummy_frame)
        print("Obstacle detector verification successful.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed during obstacle detector verification: {e}")
        print("  Check model weights path and configurations in obstacle_detector.py.")
        sim.close()
        return

    # --- 5. 主仿真循环 ---
    print("Starting main simulation loop. Press Ctrl+C in the terminal to stop.")
    frame_count = 0
    try:
        while True:
            # a. 从PyBullet虚拟相机获取当前图像帧
            # TODO: 【可配置】调整相机图像的获取尺寸，与检测器期望的输入或性能平衡
            # 较小的尺寸（如320x240）会提高整体循环速度
            sim_frame_bgr = sim.get_camera_image(width=320, height=240)
            if sim_frame_bgr is None or sim_frame_bgr.size == 0:
                print("Warning: Received empty frame from simulation camera.")
                time.sleep(0.1)  # 稍作等待
                continue

            # b. (可选) 显示从PyBullet相机获取的原始图像帧
            if connection_mode == p.DIRECT:  # 如果是无头模式，我们可能想自己显示相机内容
                cv2.imshow("PyBullet Virtual Camera", sim_frame_bgr)

            # c. 将图像帧传递给检测脚本，得到检测结果
            detections, img_h, img_w = detect_obstacles_in_image(sim_frame_bgr)

            # d. (可选) 可视化YOLOv5的检测结果 (如果需要独立的窗口显示)
            # 你可以写一个辅助函数在 obstacle_detector.py 或这里来绘制框
            # frame_with_boxes = sim_frame_bgr.copy() # 先复制一份
            # for det in detections:
            #     xmin, ymin, xmax, ymax = map(int, det['bbox'])
            #     label = f"{det['class_name']}: {det['confidence']:.2f}"
            #     cv2.rectangle(frame_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            #     cv2.putText(frame_with_boxes, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imshow("YOLOv5 Detections on Sim", frame_with_boxes)

            # e. 将检测结果传递给决策逻辑，得到动作指令
            action_command = get_action_command_simple(detections, img_w, img_h)

            # f. 打印信息（用于调试和演示）
            if frame_count % 10 == 0:  # 每10帧打印一次详细信息，避免刷屏
                print(f"\n--- Frame {frame_count} ---")
                if detections:
                    print(f"  Detections ({len(detections)}):")
                    for i, det in enumerate(detections):
                        print(
                            f"    {i + 1}. Class: {det['class_name']}, Conf: {det['confidence']:.2f}, BBox: {det['bbox']}")
                else:
                    print("  Detections: None")
                print(f"  Decision: {action_command}")

            # g. 将动作指令传递给PyBullet控制器，执行仿真动作
            sim.execute_action(action_command)

            # h. PyBullet仿真步进
            sim.step()

            # i. 控制循环频率 (可选，PyBullet的time.sleep(1./240.)通常用于物理引擎的稳定)
            # 这里我们控制整个感知-决策-行动循环的速率
            # TODO: 【可调参数】根据你的电脑性能和期望的演示流畅度调整
            time.sleep(1.0 / 30.0)  # 尝试达到约 30 FPS 的演示效果

            # j. (如果用了cv2.imshow) 处理按键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
                print("INFO: 'q' pressed, exiting simulation loop.")
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nINFO: Simulation loop interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"ERROR in main simulation loop: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的异常信息
    finally:
        # --- 6. 关闭仿真环境和资源 ---
        print("Closing PyBullet simulation...")
        sim.close()
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
        print("--- Simulation Finished ---")


if __name__ == '__main__':
    main()