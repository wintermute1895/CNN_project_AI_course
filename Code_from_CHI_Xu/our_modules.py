import torch
import torch.nn as nn

class SEBlock(nn.Module): # 对应common.py 给我们的新零件起个名字
    def __init__(self, c1, r=16):  # c1是输入通道数（有多少根电线进来），r是压缩比例
CHI_Xu/CNN_work
        super().__init__() #change

develop
        # 第1步：压缩 (Squeeze) - 把所有电线的信息汇总一下
        self.squeeze = nn.AdaptiveAvgPool2d(1) # 全局平均池化，把每个通道的特征图变成一个数
        # 第2步：激励 (Excitation) - 学习哪些电线更重要
        self.excitation = nn.Sequential(
            nn.Conv2d(c1, c1 // r, 1), # 用一个1x1的卷积（小转换器）减少电线数量
            nn.SiLU(),                 # 涂点“胶水”
            nn.Conv2d(c1 // r, c1, 1), # 再用一个1x1的卷积恢复电线数量
            nn.Sigmoid()               # 输出0-1之间的权重，表示每根电线的重要性
        )

    def forward(self, x): # 这是新零件的工作流程
        # 输入的信号 x (有很多根电线)
        weights = self.excitation(self.squeeze(x)) # 得到每根电线的重要性权重
        return x * weights # 用原始信号乘以重要性权重，重要的信号会增强，不重要的会减弱