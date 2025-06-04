本项目为天津大学自动化学院2025年 人工智能概论 课程的期末大作业，由五人小组合作完成；项目目标为在yolov5s的基础上通过改进模型结构，对特定任务如障碍物检测与运动规划进行效果优化，实现具身智能感知；以展示pybullet仿真中的完整”感知——决策——行动“逻辑链条来体现课程对”具身智能“的要求。

---
### 一、项目分工：

**A.迟旭（模型构建、决策逻辑脚本与统筹协调）**：

> 1.负责基于yolov5s改进模型结构，增加了SEBlock模块与CBAM模块，并尝试修改参数和位置进行对比测试。
> 
> 2.编写onnx可视化脚本，进行模型结构可视化展示。
> 
> 3.修改common.py和yolo.py（导入自定义模块的方式），并编写模型构建集成测试
> 
> 4.为每个新模型编写yaml（配置文件）用于后续训练
> 
> 5.编写决策逻辑脚本，用于接收检测脚本的输入，输出结果给仿真脚本
> 
> 6.每天与所有人对接工作进展，审查代码，合并PR；写文档
> 
> 7.编写.gitignore文件用于版本控制


**B.张津亮（自定义数据集搭建）**

> 1.查看yolov5使用的官方数据集，从中筛选出一部分更贴近项目目标的数据用于训练模型
> 
> 2.在kaggle和huggingface上寻找更符合我们项目目标的障碍物检测型数据集
> 
> 3.拍摄校园内的日常照片，并使用labelimg等工具进行高质量标注
> 
> 4.整合以上数据成为自定义数据集，并分别编写格式转换脚本将其转换为yolo能够识别的类型。
> 
> 5.编写数据集配置文件用于后续训练


**C.李安琪（模型训练与结果可视化）**

> 1.使用官方模型加载预训练权重在自定义数据集上训练出一个基线模型，之后的改进都与之对比
> 
> 2.使用自定义数据集配置文件与自定义模型配置文件在随机初始化权重的条件下进行训练，训练出新的不同模型
> 
> 3.编写脚本将训练结果转换为可视化对比图表进行统计分析
> 
> 4.实时监控训练过程，编写模型对比分析文档；进行超参数调优


**D.高鸣邦（检测脚本）

> 1.在yolov5官方的检测脚本的基础上编写能够获取真实图像输入、真实视频输入和能从pybullet的虚拟相机获取图像输入，并可以使用我们改进之后的模型权重文件的脚本，输出特定的检测结果
> 
> 2.辅助PPT制作与文档编写


**E.刘竞聪（仿真脚本、联调脚本与PPT制作）**

> 1.编写pybullet仿真脚本，实现对仿真模拟小车的控制和仿真环境障碍物搭建
> 
> 2.编写联调主程序，调用检测脚本、决策逻辑脚本和仿真脚本，实现封装演示仿真效果
> 
> 3.为各个脚本编写单元测试，部分测试集成到脚本内部
> 
> 4.制作PPT用于汇报演示

### 二、技术选型过程

经过学习，我们决定采用yolov5s这一成熟、轻量化的模型进行改进。尝试增加SEBlock模块和CBAM模块，使用git进行版本管理,github作为代码托管平台。使用docker封装，便于其他人运行。

### 三、CNN原理相关

首先简单地再次讲一下CNN的原理：就是通过卷积操作提取特征；每个卷积核都有自己擅长识别的“模式”，如果图片这个区域符合它的“模式”，提取出来的特征图里这个区域的数值就会比较大（可以简单用黑板或PPT举个例子）。就像一张图片是由RGB三个通道叠加而成的，用不同的卷积核计算出来很多张特征图，每张特征图叠加起来就是新的图像，然后继续做卷积操作，会不断的把浅层识别出来的小的特征组合成更加完整的，有意义的特征，就像识别一只猫，可能先识别轮廓（圆弧、直线等等），再把这些特征组合成眼睛、耳朵等等，再组合成猫脸。归一化和激活函数是为了辅助模型学习得更好，例如激活函数就像一个“阀门”，关掉特征权重不够的结果，只保留比较明显的特征，这些都是为了让模型学习的效率更高。

那么yolov5s的结构是什么样子的呢？大家可以看这张图，是我用onnx文件导出到里进行模型结构可视化的结果。从宏观角度来讲，yolov5s可以分为三部分：backbone、neck、head：

**1. Backbone (骨干网络)：从像素到特征的“炼金术”**

- **作用通俗讲：** “Backbone是模型的‘眼睛’和‘初级大脑’。它的任务是把输入的原始图片（一堆像素点）转换成计算机更容易理解的‘特征’。这些特征就像是图片的关键信息摘要，比如哪里有边缘、哪里有角点、哪些区域的纹理比较特殊等等。”
    
- **YOLOv5s (v7.0) Backbone 结构解读 (结合yolov5s_v7_original_backbone.yaml或类似的标准结构)：**
    
    - **初始下采样 (Stem - 例如第0层 Conv):**
        
        - “一开始，模型会用一个比较大的卷积核（比如6x6，步长为2）对图片进行一次快速的‘粗略扫描’和‘尺寸压缩’。这就像我们看远处的东西，先眯起眼睛大概看一下轮廓。这一步会迅速减少数据量，并提取出最基础的边缘和纹理信息。”
            
        - **输出：** 尺寸减半的特征图，通道数增加 (例如32通道)。
            
    - **多级特征提取 (Stages - 例如后续的 Conv+C3 组合)：**
        
        - “接下来，Backbone会经历多个‘阶段’。每个阶段通常包含：”
            
            - 一个 Conv 层（步长为2）：继续“压缩”特征图的空间尺寸（让模型看得更“远”，关注更大范围的上下文），同时增加通道数（让模型从更多“角度”去理解特征）。
                
            - 若干个 C3 模块：这是YOLOv5的核心‘特征加工厂’。C3模块借鉴了CSPNet的思想，结构比较精巧，它内部有很多小的卷积和Bottleneck（瓶颈）结构。它的好处是能在不大幅增加计算量的前提下，深度挖掘和提炼当前尺度的特征，并且让信息在网络中流动更顺畅，学习效率更高。”
                
        - **特征的层级性：** “经过Backbone的层层处理，我们会得到不同‘深度’的特征图：”
            
            - **P3 (例如Backbone中层4的C3输出):** 尺寸相对较大 (原图1/8)，通道数适中 (例如128)。它保留了较多的**空间细节和位置信息**，对检测小目标很有帮助。
                
            - **P4 (例如Backbone中层6的C3输出):** 尺寸中等 (原图1/16)，通道数较多 (例如256)。它在细节和语义信息之间取得了平衡。
                
            - **P5 (例如Backbone中层9的SPPF输出):** 尺寸最小 (原图1/32)，通道数最多 (例如512)。它包含了最丰富的**高级语义信息**（比如“这是一个物体的整体概念”），对检测大目标和理解场景上下文很有帮助。
                
    - **SPPF模块 (在Backbone末端)：**
        
        - “在Backbone的最后，通常会有一个SPPF模块。你可以把它想象成给模型装上了‘广角镜’和‘变焦镜’的组合。它通过不同大小的池化操作，让模型能同时关注到不同感受野的信息，把局部细节和全局上下文都‘看’到，这对于识别不同大小的物体非常有帮助。”
            
- **我们对Backbone的初步理解和思考 (引出你们的改进)：**
    
    - “我们认识到，Backbone提取的特征质量直接决定了后续检测的上限。虽然YOLOv5s的Backbone已经很高效，但我们思考，在提取这些多尺度特征的过程中，是否能让模型更‘智能’地判断哪些特征通道对我们的特定障碍物更重要呢？”
        

**2. Neck (颈部网络 - FPN+PAN)：特征的“信息高速公路”与“融合中心”**

- **作用通俗讲：** “Backbone输出了好几份不同‘清晰度’和‘理解深度’的‘地图’（P3, P4, P5特征图）。Neck部分就像一个‘信息枢纽’，它的任务是把这些地图的优点结合起来，制作出几份‘超级地图’，既有高层地图的‘战略眼光’（语义信息），又有低层地图的‘精确导航’（位置信息）。”
    
- **YOLOv5s (v7.0) Neck 结构解读：**
    
    - **FPN (Feature Pyramid Network - 自顶向下)：**
        
        - “首先，Neck会把Backbone最深层、语义最丰富的P5特征图，通过‘上采样’（Upsample，把小图放大）的方式，逐层和P4、P3特征图进行‘信息共享’（通过Concat拼接，然后用C3模块进一步融合处理）。”
            
        - “这就好比，一个经验丰富的老侦察兵（P5）把他的‘大局观’传授给在前线观察细节的新兵（P3, P4），让新兵也能理解更宏观的模式。”
            
        - **结果：** 生成了初步融合了高级语义信息的 P4_fused 和 P3_fused 特征图。
            
    - **PAN (Path Aggregation Network - 自底向上)：**
        
        - “仅仅从上往下传递信息还不够。PAN结构又增加了一条‘反向汇报’的路径。它会把刚刚在FPN中融合了语义信息的P3_fused特征图，通过‘下采样’（Conv，把大图缩小）的方式，再逐层和P4_fused、P5_fused（P5的融合版本）进行‘信息补充’。”
            
        - “这就好比，前线新兵（P3_fused）把他们观察到的最新、最精确的‘地面情况’（定位信息）汇报给后方的指挥官（P4_fused, P5_fused），让指挥官的决策更接地气。”
            
        - **结果：** 生成了最终用于检测的三个尺度的特征图：P3_detect_in, P4_detect_in, P5_detect_in。这三张图都充分融合了来自不同层级的语义和位置信息。
            
- **我们对Neck的思考 (引出你们的改进)：**
    
    - “Neck是特征融合的关键环节。我们认为，如果在这个阶段能进一步优化特征的表达，让模型更好地辨别和整合来自不同路径的有用信息，将直接惠及最终的检测头。”
        

**3. Head (检测头 - Detect层)：最终的“目标锁定与识别”**

- **作用通俗讲：** “Head部分就是‘侦察兵’最终掏出‘望远镜和目标识别器’进行精确打击的部分。它接收来自Neck的三份‘超级地图’（P3, P4, P5的最终融合特征），然后在每张地图上进行预测。”
    
- **YOLOv5s (v7.0) Head (Detect模块) 工作方式：**
    
    - “对于Neck传来的每一张特征图（例如P3_detect_in），Detect模块会用一个1x1的小卷积（可以看作是一个小型的全连接层）将特征图的每个‘格子点’（grid cell）转换成预测信息。”
        
    - **锚框 (Anchors) 的作用：** “在每个格子点上，模型会基于预设的几种不同形状和大小的‘锚框’（Anchors）来进行预测。这些锚框就像是预先画好的几个‘嫌疑框’。”
        
    - **预测内容：** “对于每个锚框，模型会预测：”
        
        1. **边界框调整：** 这个锚框需要向哪个方向移动多少、放大或缩小多少，才能正好框住物体 (box_loss与之相关)。
            
        2. **物体置信度：** 这个调整后的框里到底有没有我们要找的障碍物 (obj_loss与之相关)。
            
        3. **类别概率：** 如果有障碍物，它是我们定义的22个类别中的哪一个 (cls_loss与之相关)。
            
    - **多尺度预测：** “由于有P3, P4, P5三个尺度的输入，Detect层能在不同尺度上分别进行预测，这样就能同时检测到图片中的小、中、大各种尺寸的障碍物了。”
        
- **NMS (非极大值抑制)：** “模型可能会对同一个物体产生多个重叠的预测框。NMS就像一个‘裁判’，它会根据置信度和重叠程度，把多余的框去掉，只保留最准的那个。” (这通常在Detect模块之后，在general.py中实现)

---


**二、我们的模型修改：给“侦察兵”装上更敏锐的“感官” (注意力机制)**

“基于对YOLOv5s工作原理的理解，我们团队的核心改进集中在引入和优化‘注意力机制’，目标是让模型在复杂的视觉环境中能更有效地聚焦于关键信息，从而提升障碍物检测的性能。”

**1. 引入SEBlock (Squeeze-and-Excitation Block)**

- **我的工作与理解 
    
    - **为什么引入SEBlock：** “我们首先尝试了SEBlock。如前所述，CNN的每一层会产生很多特征通道，每个通道代表一种学到的模式。但并非所有通道在所有情况下都同等重要。SEBlock的核心思想就是让网络**学会判断哪些特征通道更重要，并给它们更高的权重。**”
        
    - **通俗原理：** （待写）
        
        - “‘Squeeze’操作：通过全局平均池化，将每个通道的特征图压缩成一个数字，代表这个通道的‘整体激活度’。”
            
        - “‘Excitation’操作：通过两个全连接层（我们用1x1卷积实现）学习这些通道激活度之间的关系，为每个通道生成一个0到1之间的‘重要性得分’或‘注意力权重’。”
            
        - “‘Scale’操作：将这些权重乘回到原始的特征通道上，实现对重要通道的增强和次要通道的抑制。”
            
    - **我的具体实现：**
        
        - “我负责在我们自定义的common.py中编写了SEBlock的PyTorch模块代码。” (可以展示你写的SEBlock类的核心__init__和forward代码片段，并解释输入channels_from_yaml_arg0和ratio参数的意义)
            
        - “然后，我通过修改模型的.yaml配置文件，将SEBlock插入到了网络的不同位置进行实验。例如，我们尝试了在Backbone的每个C3模块之后加入SEBlock，也尝试了在Neck部分的关键特征融合节点（C3模块之后）加入SEBlock。” (可以展示一个YAML修改片段，高亮SEBlock的插入)
            
        - “我们还对SEBlock的reduction_ratio (r) 参数进行了实验，例如r=8, 16, 32，以观察不同压缩比对性能和模型复杂度的影响。”
            
    - **预期与实际效果 (结合C同学的实验结果)：**
        
        - “我们预期SEBlock能够帮助模型更聚焦于与障碍物相关的特征，从而提升检测精度。”
            
        - “(展示C同学的对比表格/图表) 实验结果显示，例如，在Neck部分加入SEBlock (r=16) 的版本，相较于基线模型，在我们的自定义数据集上，mAP@0.5提升了X%，mAP@0.5:0.95提升了Y%，而参数量仅增加了Z。这初步验证了SEBlock在我们任务上的有效性。” (用真实数据替换X, Y, Z)
            

**2. 引入CBAM (Convolutional Block Attention Module)**

- **我的工作与理解
    
    - **为什么引入CBAM：** “在SEBlock只关注通道维度的基础上，我们进一步探索了CBAM。CBAM的优势在于它**同时引入了通道注意力和空间注意力**，不仅告诉模型‘关注哪些特征类型（通道）’，还告诉模型‘关注图像的哪些区域（空间）’。”
        
    - **通俗原理：** (回顾“频道调音师”+“聚光灯操作员”比喻)
        
        - **通道注意力模块 (CAM)：** “CBAM的CAM与SEBlock类似，但它同时使用了平均池化和最大池化来捕捉更丰富的通道统计信息，然后通过一个共享的MLP（多层感知机，用1x1卷积实现）来生成通道权重。”
            
        - **空间注意力模块 (SAM)：** “在通道权重被应用之后，SAM开始工作。它会沿着特征图的通道维度进行平均池化和最大池化，得到两张代表空间信息的特征图。然后将这两张图拼接起来，通过一个卷积层（通常是7x7）来学习一个空间注意力图，这个图会高亮显示那些对任务重要的空间区域。”
            
        - **顺序应用：** “CBAM将CAM和SAM顺序应用，实现对特征在通道和空间维度上的双重增强。”
            
    - **我的具体实现：**
        
        - “我同样在自定义的common.py中编写了CBAM模块及其子模块ChannelAttention和SpatialAttention的PyTorch代码。” (可以展示CBAM类或其子模块的核心代码片段，并解释channels_from_yaml_arg0, ratio, kernel_size参数)
            
        - “我们主要实验了将CBAM放置在Neck部分，替代之前SEBlock的位置，或者与SEBlock进行组合（如果时间允许）。” (展示YAML修改片段)
            
    - **预期与实际效果 (结合C同学的实验结果)：**
        
        - “我们期望CBAM能够通过其更全面的注意力机制，在SEBlock的基础上进一步提升模型的性能，尤其是在目标的精确定位和复杂背景下的识别能力。”
            
        - “(展示C同学的对比表格/图表，包含CBAM版本的结果) 实验结果显示，CBAM版本模型在...(例如mAP@0.5:0.95上表现突出，或者在某些特定难检测类别上效果更好)...。虽然CBAM的参数量和计算量比SEBlock略高，但...(根据结果说明是否值得)。”
            

**3. YAML配置与模型构建的理解与实践**

- **我的工作与理解 ：**
    
    - “为了实现这些网络结构的修改，我深入学习了YOLOv5通过YAML配置文件来定义模型架构的方式。”
        
    - **关键点：**
        
        - “[from, number, module, args] 这四个参数的精确含义和相互关系，特别是from索引在插入/删除模块后的正确更新。”
            
        - “depth_multiple和width_multiple如何影响实际的网络深度和通道数。”
            
        - “自定义模块（如SEBlock, CBAM）的__init__方法参数如何与YAML中args列表进行对应，特别是parse_model对已知模块和未知模块的参数传递方式的差异。”
            
    - **实践：** “我编写并使用了test_model_build.py脚本来频繁验证我修改后的YAML文件是否能够成功构建模型并通过前向传播，这帮助我们快速定位和修复了许多由于索引错误或参数不匹配导致的问题。”
        
    - “我还通过sys.path的修改，实现了在不‘污染’原始YOLOv5源码的情况下，让模型构建过程优先加载我们自定义的common.py文件，这保证了我们代码的整洁性和可管理性。”
        

---
### 四、训练过程与具身智能仿真模拟

在项目推进过程中，我们发现在真实硬件上进行开发遇到了很多问题，例如Realsense系列的SDK使用，机器人运动控制等等难度较高，因此我们将目标转向先用pybullet实现仿真模拟；同时负责数据与训练的同学通过修改配置文件、重新训练，让自定义模型在原有的真实图像输入检测能力下降较少的条件下提高了对pybullet内置障碍物的检测能力，以下是我们100个epoch下不同模型的数据对比，通过100个epoch下的训练我们能够初步筛选出具有提升潜力的模型，进行进一步训练，最终得到的权重将用于检测脚本。

训练工作基本结束后，我们开始 检测脚本、决策逻辑脚本与仿真模拟脚本 三项同步推进，并开始编写联调主程序

联调成果：可以实现小车在pybullet环境内进行运动避障和实时环境检测。

---
## 五、项目复现 (写给读者)

本项目提供了两种复现方式：通过 Docker (推荐，环境一致性好) 或直接在本地环境搭建。

### 方式一：使用 Docker (推荐)

使用 Docker 可以确保在与开发者一致的环境中运行项目，避免因环境差异导致的问题。

1.  **环境准备:**
    *   **安装 Docker:** 请根据你的操作系统从 Docker 官网下载并安装 Docker Desktop (Windows/macOS) 或 Docker Engine (Linux)。
    *   **(可选，推荐用于GPU加速训练/推理) NVIDIA GPU 用户:**
        *   安装最新的 NVIDIA 驱动。
        *   安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，以便 Docker 容器可以使用 GPU。

2.  **获取项目文件:**
    *   下载本项目的压缩包并解压，或者通过 `git clone` 获取项目。

3.  **准备数据集 (如果需要重新训练或使用特定数据集进行评估/演示):**
    *   在你的项目根目录下创建一个用于存放数据集的文件夹，例如 `my_datasets/`。
    *   将你的数据集（例如，COCO格式的图片和标注文件，或已经转换为YOLO格式的数据）放入此文件夹。
    *   **注意:** 项目中使用的配置文件（如数据 `.yaml` 文件和模型 `.yaml` 文件）中的路径可能需要根据你实际挂载到容器内的数据集路径进行调整。我们推荐将宿主机的数据集目录挂载到容器内的 `/app/datasets`。

4.  **构建 Docker 镜像:**
    *   打开终端或命令行，导航到解压后的项目根目录 (包含 `Dockerfile` 文件的目录)。
    *   运行以下命令构建 Docker 镜像 (将 `your-custom-tag`替换为你想要的标签，如 `latest` 或 `v1.0`):
        ```bash
        docker build -t yolov5-tju-ai:your-custom-tag .
        ```
    *   构建过程可能需要一些时间，因为它会下载基础镜像并安装所有依赖。

5.  **运行 Docker 容器并执行任务:**

    我们推荐启动一个交互式的 Bash Shell，这样你可以在容器内灵活执行各种脚本。

    *   **启动交互式 Shell (通用，可根据需求添加 GPU 和 GUI 支持):**
        ```bash
        # 基础命令 (无 GPU, 无 GUI)
        docker run -it --rm \
            -v $(pwd)/my_datasets:/app/datasets \         # 将你的本地数据集目录挂载到容器的 /app/datasets
            -v $(pwd)/runs_output:/app/yolov5/runs \    # 将本地 runs_output 目录挂载到容器的 /app/yolov5/runs (用于保存训练结果)
            yolov5-tju-ai:your-custom-tag \
            bash

        # Linux 用户，如果需要 PyBullet GUI (X11 转发):
        # xhost +local:docker
        # docker run -it --rm \
        #     --gpus all \                                  # 如果有 NVIDIA GPU 并安装了 NVIDIA Container Toolkit
        #     --env="DISPLAY" \
        #     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        #     -v $(pwd)/my_datasets:/app/datasets \
        #     -v $(pwd)/runs_output:/app/yolov5/runs \
        #     yolov5-tju-ai:your-custom-tag \
        #     bash

        # macOS 用户 (需要 XQuartz 并配置, 如果需要 PyBullet GUI):
        # IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}') # 获取 IP
        # xhost + $IP
        # docker run -it --rm \
        #     --env DISPLAY=$IP:0 \
        #     -v /tmp/.X11-unix:/tmp/.X11-unix \
        #     -v $(pwd)/my_datasets:/app/datasets \
        #     -v $(pwd)/runs_output:/app/yolov5/runs \
        #     yolov5-tju-ai:your-custom-tag \
        #     bash
        # (如果 macOS 使用 GPU, 可能需要更复杂的 Docker Desktop 配置或特定基础镜像)

        # Windows 用户 (需要 VcXsrv 或 Xming 并配置, 如果需要 PyBullet GUI):
        # docker run -it --rm \
        #     --gpus all \                                  # 如果有 NVIDIA GPU (需要 Docker Desktop WSL2 后端和 NVIDIA Container Toolkit 支持)
        #     -e DISPLAY=<你的Windows主机IP>:0.0 \
        #     -v ${PWD}/my_datasets:/app/datasets \
        #     -v ${PWD}/runs_output:/app/yolov5/runs \
        #     yolov5-tju-ai:your-custom-tag \
        #     bash
        ```
        *   **说明:**
            *   `-it`: 以交互模式运行并分配一个伪终端。
            *   `--rm`: 容器退出时自动删除。
            *   `-v $(pwd)/my_datasets:/app/datasets`: 将当前宿主机目录下的 `my_datasets` 文件夹挂载到容器内的 `/app/datasets` 路径。请确保 `my_datasets` 存在或替换为你的实际数据集路径。
            *   `-v $(pwd)/runs_output:/app/yolov5/runs`: 将当前宿主机目录下的 `runs_output` 文件夹挂载到容器内的 `/app/yolov5/runs` 路径。训练产生的结果（模型权重、日志等）会保存在这里。请确保 `runs_output` 存在。
            *   `--gpus all`: (NVIDIA GPU 用户) 允许容器访问所有可用的 GPU。
            *   GUI 相关参数 (`-e DISPLAY`, `-v /tmp/.X11-unix...`): 用于在容器内运行的图形化应用（如 PyBullet 仿真窗口）显示在宿主机上。具体配置因操作系统而异。
            *   `yolov5-tju-ai:your-custom-tag`: 你构建的镜像名称和标签。
            *   `bash`: 在容器内启动一个 Bash Shell。

    *   **在容器内执行脚本:**
        成功进入容器后，你会看到类似 `root@<container_id>:/app#` 的提示符。现在你可以像在本地一样运行脚本了。
        ```bash
        # (在容器内 /app 目录下)

        # 示例 1: 数据格式转换 (如果你的数据集需要)
        # 假设你的原始数据在 /app/datasets/raw_data，转换后输出到 /app/datasets/yolo_data
        # python cocotoyolo.py --input_dir /app/datasets/raw_data --output_dir /app/datasets/yolo_data
        # (请根据你的 cocotoyolo.py 脚本实际参数进行调整)

        # 示例 2: 开始训练
        # 确保你的数据配置文件 (例如 my_custom_data.yaml) 中的路径指向容器内的 /app/datasets/...
        # 并且你的模型配置文件 (例如 Code_from_CHI_Xu/yolov5s_with_se.yaml) 也被正确引用
        python yolov5/train.py \
            --img 640 \
            --batch 16 \
            --epochs 100 \
            --data /app/datasets/your_data_config.yaml \
            --cfg Code_from_CHI_Xu/yolov5s_your_custom_model.yaml \
            --weights yolov5s.pt \ # 使用项目根目录下的预训练权重
            --project /app/yolov5/runs/train \ # 训练输出将保存到挂载的宿主机 runs_output 目录
            --name my_docker_experiment

        # 示例 3: 验证模型
        # 使用你训练得到的权重，它现在应该在 /app/yolov5/runs/train/my_docker_experiment/weights/best.pt
        python yolov5/val.py \
            --weights /app/yolov5/runs/train/my_docker_experiment/weights/best.pt \
            --data /app/datasets/your_data_config.yaml \
            --img 640 \
            --task test # 或其他任务参数

        # 示例 4: 运行主仿真循环 (演示)
        # 假设使用训练好的最佳权重
        python main_simulation_loop.py --weights /app/yolov5/runs/train/my_docker_experiment/weights/best.pt
        # 如果 main_simulation_loop.py 默认启动 PyBullet GUI，请确保你运行容器时添加了 GUI 转发参数。
        # 如果你的脚本支持 --no-gui 或类似参数用于非图形化运行，也可以使用。

        # 退出容器
        # exit
        ```

### 方式二：本地环境搭建 (不使用 Docker)

如果你希望直接在本地环境运行项目：

1.  **获取项目文件:**
    *   下载本项目的压缩包并解压，或者通过 `git clone` 获取项目。

2.  **创建 Python 虚拟环境 (推荐):**
    我们强烈建议使用虚拟环境 (如 venv 或 conda) 来隔离项目依赖。假设你使用 Python 3.9+ (与 Dockerfile 中一致)。
    ```bash
    # 使用 venv (Python 内置)
    python -m venv venv_yolov5_tju
    # 激活环境
    # Windows:
    # venv_yolov5_tju\Scripts\activate
    # Linux/macOS:
    # source venv_yolov5_tju/bin/activate

    # 或者使用 conda
    # conda create -n yolov5_tju_env python=3.9 -y
    # conda activate yolov5_tju_env
    ```

3.  **安装依赖:**
    进入项目根目录，安装 `requirements.txt` 中列出的依赖：
    ```bash
    pip install -r requirements.txt
    ```
    *   **注意:** `requirements.txt` 是为 Docker (Linux) 环境生成的。在 Windows 或 macOS 上，某些包的安装可能需要额外的步骤或有细微差异。如果遇到问题，请尝试单独安装出问题的包，并查找其特定于你操作系统的安装指南。PyTorch 的安装尤其建议参考其官网根据你的 CUDA 版本（如果使用 GPU）或 CPU 来获取正确的安装命令。

4.  **准备数据集和预训练权重:**
    *   将你的数据集放置在项目中合适的位置，并确保配置文件中的路径正确指向它们。
    *   确保预训练权重文件 (如 `yolov5s.pt`) 位于项目根目录或脚本可以找到的位置。

5.  **执行项目脚本:**
    现在你可以按照项目原本的流程运行脚本：
    *   **数据格式转换 (如果需要):**
        ```bash
        python cocotoyolo.py # (根据脚本实际参数调整)
        ```
    *   **开始训练:**
        ```bash
        python yolov5/train.py --img 640 --batch 16 --epochs 100 --data path/to/your_data_config.yaml --cfg path/to/your_model_config.yaml --weights yolov5s.pt --name local_experiment
        ```
    *   **验证模型:**
        ```bash
        python yolov5/val.py --weights yolov5/runs/train/local_experiment/weights/best.pt --data path/to/your_data_config.yaml --img 640
        ```
    *   **运行主仿真循环 (演示):**
        ```bash
        python main_simulation_loop.py --weights yolov5/runs/train/local_experiment/weights/best.pt
        ```
        (确保 PyBullet 可以正常显示 GUI，或者你的脚本支持非 GUI 模式)。

    *   **查看屏幕上的 PyBullet 界面，小车开始运动。**

---

**重要提示给使用者：**

*   **路径配置：** 无论是使用 Docker 还是本地环境，请务必检查并根据你的实际情况修改项目中的**数据配置文件 (`.yaml`)** 和**模型配置文件 (`.yaml`)** 中的路径，确保它们指向正确的数据集位置和模型定义。
*   **权重文件：** 运行推理或仿真时，确保 `--weights` 参数指向正确的模型权重文件 (`.pt`)。
*   **GPU 支持：** 如果你希望使用 GPU 加速，请确保你的环境（本地或 Docker）已正确配置 CUDA 和 cuDNN (对于 NVIDIA GPU)。

### 六、我们遇到过的问题：

1.基线模型在35个epoch就已经表现良好：使用了预训练权重、并且数据集当时尚未完善，只有官方数据集

2.有三次自定义模型实验数据重合严重：训练的同学忘记改配置文件路径，得到了几次无效的训练结果

3.部分自定义模型在100个epoch之后表现不佳：一方面是数据集质量还不够高，另一方面是因为预训练权重因为模型结构更改后无法匹配，只能默认从随机初始化开始训练，效率较低

4.模型构建的问题：最理想的方式其实是不对官方原生的common.py做侵入式修改，但我们改了，比较直观方便，容易通过构建测试

5.不同协作者的yolov5版本不同：用git看版本信息，找commit hash同步，或直接复制一个人的项目文件（最简单粗暴的同步方式）。

6.复杂冲突无法合并PR：没啥说的，耐心地看、改。

7.联调初期遇到的问题：大部分是因为不同脚本之间用到的工具来源的版本不一致，有些和yolo有关，有些和pybullet有关，同步版本，详细核对数据接口就好了。

### 未来的目标：

应用于实际的具身智能场景，比如智能驾驶之类，或者先从简单的来：给电控小车加一个摄像头，用来进行目标检测，智能避障。

---
### 结语：

感谢各位队友、老师，感谢Gemini和Deep Seek。

一周之内我一个人和Gemini 2.5 pro的对话量就已经达到了恐怖的150wtoken，聊崩了三个对话；科技改变生活，根据Gemini自己评估，如果没有ai帮助，我们这个难度的大作业可能要写8~16周才能完成，而现在我们作为自动化专业的学生，在短期内掌握这么多工具和技能，走完了一遍完整的技术选型、模型训练、实际应用的过程其实也是一件挺不容易的事儿。谢谢你们！深度学习真的很好玩。


---
#### 附录：

每个人的代码贡献：
