import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from glob import glob


def csv_to_plot(csv_path, output_path=None):
    """将CSV文件转换为图表"""
    try:
        # 读取CSV
        data = pd.read_csv(csv_path)

        # 自动检测列名（兼容不同格式）
        step_col = None
        value_col = None

        # 打印列名用于调试
        print(f"  列名: {data.columns.tolist()}")

        # 查找step列
        for col in ['Step', 'step', 'iteration', 'epoch']:
            if col in data.columns:
                step_col = col
                break

        # 查找value列
        for col in ['Value', 'value', 'metric_value', 'loss', 'accuracy']:
            if col in data.columns:
                value_col = col
                break

        # 如果没找到，尝试使用前两列
        if not step_col and len(data.columns) >= 2:
            step_col = data.columns[0]
            value_col = data.columns[1]
            print(f"  使用默认列: {step_col} 和 {value_col}")

        if not step_col or not value_col:
            print(f"  ✗ 缺少必需的列，跳过")
            return

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(data[step_col], data[value_col], label=value_col)
        plt.xlabel(step_col)
        plt.ylabel(value_col)

        # 从文件名提取标题
        title = os.path.splitext(os.path.basename(csv_path))[0]
        plt.title(title)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图片
        if output_path is None:
            output_path = os.path.splitext(csv_path)[0] + '.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 已保存: {output_path}")

    except Exception as e:
        print(f"  ✗ 处理时出错: {str(e)}")


def batch_convert(folder_path, output_folder=None):
    """批量转换文件夹中的CSV文件"""

    # 检查输入文件夹
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return

    # 创建输出文件夹
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 查找CSV文件
    csv_files = glob(os.path.join(folder_path, '*.csv'))

    if not csv_files:
        print(f"在 '{folder_path}' 中没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")
    print("-" * 50)

    # 批量处理
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] 处理: {os.path.basename(csv_file)}")

        if output_folder:
            output_path = os.path.join(
                output_folder,
                os.path.basename(csv_file).replace('.csv', '.png')
            )
        else:
            output_path = None

        csv_to_plot(csv_file, output_path)

    print("-" * 50)
    print("批量转换完成！")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python batch_csv_to_img.py <输入文件夹> [输出文件夹]")
        print("\n示例:")
        print("  python batch_csv_to_img.py exported_csvs")
        print("  python batch_csv_to_img.py exported_csvs output_images")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None

    batch_convert(folder_path, output_folder)

# python batch_csv_to_img.py se_backbone_r32_csvs se_backbone_r32_images

