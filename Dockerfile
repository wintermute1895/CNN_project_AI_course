# 1. 选择 Python 3.12 基础镜像
FROM python:3.12-slim

# 2. 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安装系统依赖
# 根据 requirements.txt (特别是 opencv-python, pybullet, matplotlib, torch) 推断
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    # PyBullet GUI 和 OpenCV 需要的库
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # ffmpeg for OpenCV video capabilities
    ffmpeg \
    # graphviz (for diagrams, though likely not needed at runtime in Docker unless you generate them there)
    graphviz \
    # 清理 apt 缓存以减小镜像体积
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. 设置工作目录
WORKDIR /app

# 5. 复制 requirements.txt 并安装 Python 包
# !!! 重要: 请确保你的 requirements.txt 文件只包含标准的包名和版本号，
# 例如 "torch==2.5.1"，而不是 "torch @ file:///..." 这种本地路径。
# 如果你的 requirements.txt 包含本地文件路径，你需要先清理它。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 6. 复制项目代码和必要的配置文件
# 优先复制 yolov5 目录，因为它可能包含基础依赖
COPY yolov5/ ./yolov5/

# 复制 Code_from_CHI_Xu 中的模型定义和必要的 Python 文件
COPY Code_from_CHI_Xu/attention_blocks.py ./Code_from_CHI_Xu/
COPY Code_from_CHI_Xu/common_v2_SEBlock_and_CBAM.py ./Code_from_CHI_Xu/
COPY Code_from_CHI_Xu/common_version1_SEBlock.py ./Code_from_CHI_Xu/
COPY Code_from_CHI_Xu/yolo_version1_SEBlocks.py ./Code_from_CHI_Xu/
COPY Code_from_CHI_Xu/yolo_version2_CBAM_and_SEBlock.py ./Code_from_CHI_Xu/
# 复制 YAML 模型配置文件
COPY Code_from_CHI_Xu/*.yaml ./Code_from_CHI_Xu/

# 复制根目录下的核心 Python 脚本和必要的模型文件
COPY decision_maker.py ./
COPY export_custom_model_to_onnx.py ./
COPY Filter.py ./
COPY LICENSE ./
COPY main_simulation_loop.py ./
# COPY my_pybullet_test.py ./ # 测试脚本，可选
COPY obstacle_detector.py ./
COPY pybullet_sim.py ./
# COPY pybullet_test.py ./ # 测试脚本，可选
COPY README.md ./ # 包含 README 以供参考
# COPY test01.py ./ # 测试脚本，可选
COPY test_model_build.py ./ # 模型构建测试脚本，可能有用
COPY yolov5s.pt ./ # 假设这是基础预训练权重

# 复制你修改过的 common.py 和 yolo.py (如果它们在根目录)
# 根据你的 README，你修改了 common.py 和 yolo.py 来导入自定义模块。
# 你需要确定这些修改后的文件在哪里。
# 假设你的自定义模块导入路径是基于项目根目录的。
# 如果你修改了 yolov5/models/common.py 和 yolov5/models/yolo.py，
# 并且你的自定义模块 (如 SEBlock, CBAM) 在 Code_from_CHI_Xu/ 或根目录，
# 你需要确保 Python 的 import 路径能够找到它们。
# 一种方式是将 Code_from_CHI_Xu 也加入 PYTHONPATH，或者确保你的导入语句正确。
# 例如，如果 common_v2_SEBlock_and_CBAM.py 在 Code_from_CHI_Xu/
# 并且 yolov5/models/common.py 尝试 from Code_from_CHI_Xu.common_v2_SEBlock_and_CBAM import SEBlock，
# 这就需要 Code_from_CHI_Xu 目录在 Python 的搜索路径中。
# 或者你的导入是相对的，例如从 `..Code_from_CHI_Xu` (如果结构允许)

# 解决自定义模块导入问题的一个简单方法是将相关自定义模块也放在 yolov5/models/ 下，
# 或者调整 PYTHONPATH
# ENV PYTHONPATH "${PYTHONPATH}:/app/Code_from_CHI_Xu" # 示例，如果需要

# 复制 labels 目录 (如果它包含运行时需要的文件，如 classes.txt，并且体积不大)
COPY labels/ ./labels/

# 如果你的自定义模型 YAML 文件 (在 Code_from_CHI_Xu 中) 引用了 yolov5/models 中的标准 YAML，
# 那么 yolov5/models/ 目录也需要被正确复制 (上面已复制整个 yolov5/)

# 7. (可选) 设置默认的容器启动命令
# 推荐使用 bash，以便灵活执行不同任务
CMD ["bash"]