# generate_diagram_with_diagrams_simple_fixed.py (简化标签版)
from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom


def create_project_diagram_diagrams_simplified_labels(filename="project_arch_diagrams_simple_labels", direction="LR"):
    graph_attrs = {
        "fontsize": "10",
        "splines": "ortho",
    }
    node_attrs = {
        "shape": "box",
        "style": "rounded,filled",
        "fillcolor": "lightblue",
        "fontname": "Arial",  # 尝试一个常见且你系统上肯定有的字体
        "fontsize": "9"
    }
    edge_attrs = {
        "fontname": "Arial",
        "fontsize": "8"
    }

    with Diagram(filename, show=False, direction=direction, graph_attr=graph_attrs, node_attr=node_attrs,
                 edge_attr=edge_attrs):
        # --- 大幅简化标签 ---
        data_prep = Custom("数据准备 (B)", "")
        training = Custom("模型训练 (C, A)", "")
        best_model = Custom("最佳模型", "")
        input_source = Custom("输入源", "")

        with Cluster("感知模块 (D)"):
            detector_core = Custom("YOLOv5 推理", "")
            detection_output = Custom("检测结果", "")
            detector_core >> detection_output

        decision_logic = Custom("决策逻辑 (A)", "")
        decision_log = Custom("决策日志", "")

        with Cluster("行动模块: PyBullet (E, A)"):
            pybullet_env_setup = Custom("PyBullet环境", "")  # 注意：之前这里写的是 "PyBullet环境与机器人初始化"
            virtual_camera = Custom("虚拟相机", "")  # 之前是 "虚拟相机图像获取"
            robot_action_execution = Custom("机器人动作", "")  # 之前是 "机器人动作执行"

        simulation_result = Custom("仿真结果", "")

        data_prep >> training
        training >> best_model
        input_source >> detector_core
        best_model >> detector_core
        detection_output >> decision_logic
        decision_logic >> robot_action_execution
        decision_logic >> decision_log
        robot_action_execution >> simulation_result
        virtual_camera >> Edge(label="虚拟相机帧\n(闭环)", style="dashed", color="blue") >> detector_core  # 边的标签也可以简化

    print(f"SUCCESS: Diagram saved as {filename}.png")


if __name__ == '__main__':
    create_project_diagram_diagrams_simplified_labels()