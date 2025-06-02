# generate_project_diagram.py

from graphviz import Digraph


def create_project_diagram(filename="project_architecture", file_format="png"):
    """
    使用 Graphviz 生成项目结构图。
    """
    # 创建一个有向图 (Digraph)
    # 'LR' 表示从左到右布局，也可以用 'TB' (从上到下)
    dot = Digraph(comment='YOLOv5 Embodied AI Project Architecture', graph_attr={'rankdir': 'LR', 'splines': 'ortho'})

    # 定义图的全局节点和边的属性 (可选)
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # --- 定义核心模块节点 ---
    # 可以为每个节点指定标签、形状、颜色等属性
    dot.node('A', '数据准备与增强\n(B同学)\n- 数据集 (图像/视频)\n- your_dataset.yaml\n- 增强策略 (hyp.yaml)')
    dot.node('B',
             'CNN模型训练与选择\n(C同学, A同学指导)\n- YOLOv5 (原始/SE/CBAM)\n- PyTorch, train.py, val.py\n- TensorBoard分析')
    dot.node('C', '最佳模型权重\n(best_model.pt\n+ best_model.yaml)')

    dot.node('D', '视频/图片序列输入\n(演示数据源)')
    # 对于检测模块，我们可以用一个“集群”来表示其内部组件
    with dot.subgraph(name='cluster_detector') as sub_detector:
        sub_detector.attr(label='感知模块: 障碍物检测 (D同学)', style='filled', color='lightgrey', fontname='Arial',
                          fontsize='12')
        sub_detector.node('D1', '模型加载\n(ObstacleDetector)')
        sub_detector.node('D2', '图像预处理')
        sub_detector.node('D3', 'YOLOv5推理')
        sub_detector.node('D4', '后处理 (NMS, 坐标还原)')
        sub_detector.node('D5', '输出: detections_list\n(类别, bbox, 置信度)')
        # 定义子图内部的流程
        sub_detector.edge('D1', 'D3')
        sub_detector.edge('D2', 'D3')
        sub_detector.edge('D3', 'D4')
        sub_detector.edge('D4', 'D5')

    dot.node('E', '决策逻辑模块\n(A同学)\n- decision_maker.py\n- 避障规则')

    # 对于PyBullet仿真模块，也可以用集群
    with dot.subgraph(name='cluster_simulation') as sub_simulation:
        sub_simulation.attr(label='行动模块: PyBullet仿真 (E同学, A同学支持)', style='filled', color='lightgrey',
                            fontname='Arial', fontsize='12')
        sub_simulation.node('F1', '仿真环境初始化\n(场景, 机器人, 障碍物)')
        sub_simulation.node('F2', '虚拟相机图像获取')
        sub_simulation.node('F3', '机器人动作执行\n(根据指令)')
        sub_simulation.node('F4', 'PyBullet物理引擎\n(p.stepSimulation)')
        # 定义子图内部的流程 (F2->...->F3->F4->F2 形成循环)
        # F4 指向 F2 表示下一个仿真周期的图像

    dot.node('G', '仿真演示结果\n(机器人避障行为)')
    dot.node('H', '决策时间序列日志\n(decision_log.json)')  # 你提到的输出

    # --- 定义数据流和控制流边 ---
    dot.edge('A', 'B', label='数据集\nYAML配置')
    dot.edge('B', 'C', label='训练好的模型')

    dot.edge('D', 'D2', label='图像/视频帧')  # 视频输入到预处理
    dot.edge('C', 'D1', label='模型权重\nYAML配置')  # 模型加载

    # 检测模块的整体输入可以指向 D2 (预处理接收原始帧)
    # 检测模块的整体输出是 D5
    dot.edge('D5', 'E', label='检测结果\n(detections_list,\nimg_dims)')

    dot.edge('E', 'F3', label='动作指令\n(action_command)')
    dot.edge('E', 'H', label='记录决策')  # 决策逻辑输出时间序列日志

    dot.edge('F3', 'F4', label='控制机器人')
    dot.edge('F4', 'G', label='更新仿真状态')

    # 可选的闭环：仿真相机图像 -> 检测模块
    # 如果你想强调这个闭环，可以添加这条边
    # 从 F2 (虚拟相机图像获取) 指向 D2 (图像预处理)
    dot.edge('F2', 'D2', label='虚拟相机帧 (闭环)', style='dashed', color='blue', constraint='false')
    # constraint='false' 尝试避免这条边过多影响主布局方向
    # 你也可以不加这条边，在讲解时口头说明闭环

    # --- 渲染并保存图形 ---
    try:
        dot.render(filename, format=file_format, view=False, cleanup=True)
        print(f"SUCCESS: Project diagram saved as {filename}.{file_format}")
    except Exception as e:
        print(f"ERROR: Could not render diagram. Is Graphviz installed and in PATH? Error: {e}")
        print("Please ensure Graphviz (the software, not just the Python package) is installed ")
        print("and its 'bin' directory is added to your system's PATH environment variable.")
        print("Download from: https://graphviz.org/download/")


if __name__ == '__main__':
    # 你可以指定输出文件名和格式
    create_project_diagram(filename="yolov5_embodied_ai_architecture", file_format="png")
    # create_project_diagram(filename="yolov5_embodied_ai_architecture", file_format="pdf")
    # create_project_diagram(filename="yolov5_embodied_ai_architecture", file_format="svg")