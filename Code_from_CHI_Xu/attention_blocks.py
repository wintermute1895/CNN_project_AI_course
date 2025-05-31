# 文件: CNN_project_AI_course/custom_cnn_modules/attention_blocks.py
import torch
import torch.nn as nn
# 你可能需要从YOLOv5的工具中导入LOGGER，如果SEBlock内部的warning想用它
# from yolov5.utils.general import LOGGER # 假设yolov5在上一级目录

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    A channel-wise attention mechanism that adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, c1, r=16):  # c1: input channels, r: reduction ratio
        super(SEBlock, self).__init__()
        # 确保缩减后的通道数至少为1
        inter_channels = c1 // r
        if inter_channels == 0:
            inter_channels = 1 # 最小中间通道数为1
            # 可选：如果想在r过大时给出警告，需要确保LOGGER可用或使用print
            # if c1 > 1 and r > c1:
            #     try:
            #         from yolov5.utils.general import LOGGER # 尝试导入
            #         LOGGER.warning(f"SEBlock reduction ratio r={r} too large for c1={c1}. Using r resulting in 1 inter_channel.")
            #     except ImportError:
            #         print(f"Warning: SEBlock reduction ratio r={r} too large for c1={c1}. Using r resulting in 1 inter_channel.")


        self.squeeze = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.excitation = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0, bias=False), # 1x1 卷积降维
            nn.SiLU(),  # 使用SiLU激活函数，与YOLOv5风格一致
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0, bias=False), # 1x1 卷积升维
            nn.Sigmoid()  # Sigmoid输出通道权重
        )

    def forward(self, x):
        weights = self.excitation(self.squeeze(x))
        return x * weights