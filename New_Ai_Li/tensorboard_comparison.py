import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_tensorboard_csvs(folder_path, output_folder=None):
    """
    读取文件夹中的所有CSV文件，并将相同列的数据绘制在同一张图上
    """

    # 获取文件夹中所有的CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv_files.sort()

    if len(csv_files) == 0:
        print("文件夹中没有找到CSV文件！")
        return

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for i, f in enumerate(csv_files):
        print(f"  {i + 1}. {f}")

    # 读取所有CSV文件
    dataframes = {}
    all_columns = set()

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            df = pd.read_csv(file_path)
            file_name = os.path.splitext(csv_file)[0]
            dataframes[file_name] = df
            all_columns.update(df.columns.tolist())
            print(f"✓ 读取文件: {csv_file}, 形状: {df.shape}")
        except Exception as e:
            print(f"✗ 读取文件 {csv_file} 失败: {e}")

    print(f"\n成功读取 {len(dataframes)} 个文件")

    # 确定要跳过的列和x轴列
    skip_columns = ['Step', 'Wall time', 'step', 'Wall Time']
    x_column = None

    # 优先使用epoch作为x轴
    epoch_column = None
    for col in all_columns:
        if 'epoch' in col.lower():
            epoch_column = col
            x_column = epoch_column
            print(f"使用 '{col}' 作为x轴")
            break

    # 获取所有数据列（需要绘图的列）
    data_columns = [col for col in all_columns if col not in skip_columns and col != epoch_column]
    print(f"\n找到 {len(data_columns)} 个数据列需要绘制")

    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(folder_path, 'comparison_plots_fixed')

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 设置更多样化的线型和颜色
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', '#00CED1', '#FF1493',
              '#FFD700', '#4B0082', '#00FA9A', '#DC143C', '#7CFC00', '#FF69B4', '#1E90FF']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*']

    # 为每个数据列创建对比图
    for column in data_columns:
        print(f"\n正在处理列: {column}")

        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 8))

        # 记录绘制的数据信息
        plot_info = []

        # 绘制每个文件的数据
        for idx, (file_name, df) in enumerate(dataframes.items()):
            if column in df.columns:
                try:
                    # 获取数据并转换为数值类型
                    if x_column and x_column in df.columns:
                        x_data = pd.to_numeric(df[x_column], errors='coerce').values
                        y_data = pd.to_numeric(df[column], errors='coerce').values
                    else:
                        x_data = np.arange(len(df))
                        y_data = pd.to_numeric(df[column], errors='coerce').values

                    # 移除NaN值
                    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                    x_data = x_data[valid_mask]
                    y_data = y_data[valid_mask]

                    if len(y_data) > 0:
                        # 选择颜色和线型
                        color = colors[idx % len(colors)]
                        linestyle = line_styles[idx % len(line_styles)]

                        # 绘制线条
                        line = ax.plot(x_data, y_data,
                                       label=file_name,
                                       color=color,
                                       linewidth=2.5,
                                       alpha=0.9,
                                       linestyle=linestyle)[0]

                        # 每隔几个点添加标记（避免太密集）
                        marker_indices = np.linspace(0, len(x_data) - 1, min(20, len(x_data)), dtype=int)
                        ax.plot(x_data[marker_indices], y_data[marker_indices],
                                markers[idx % len(markers)],
                                color=color,
                                markersize=6,
                                alpha=0.7,
                                markeredgecolor='white',
                                markeredgewidth=1)

                        # 记录信息
                        plot_info.append({
                            'name': file_name,
                            'points': len(y_data),
                            'min': np.min(y_data),
                            'max': np.max(y_data),
                            'mean': np.mean(y_data)
                        })

                        print(
                            f"  ✓ {file_name}: 绘制了 {len(y_data)} 个数据点 (min={np.min(y_data):.4f}, max={np.max(y_data):.4f})")
                    else:
                        print(f"  - {file_name}: 没有有效数据点")
                except Exception as e:
                    print(f"  ✗ {file_name}: 处理数据时出错 - {str(e)}")
            else:
                print(f"  - {file_name}: 不包含列 '{column}'")

        if len(plot_info) == 0:
            print(f"  ! 警告: 没有文件包含列 '{column}' 的有效数据")
            plt.close()
            continue

        # 设置图形属性
        ax.set_title(f'{column} Comparison ({len(plot_info)} curves)', fontsize=16, fontweight='bold')
        ax.set_xlabel(x_column if x_column else 'Index', fontsize=12)
        ax.set_ylabel(column, fontsize=12)

        # 优化图例 - 如果曲线较多，使用两列显示
        if len(plot_info) > 5:
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                               framealpha=0.95, fontsize=10,
                               fancybox=True, shadow=True, ncol=1)
        else:
            legend = ax.legend(loc='best', framealpha=0.95, fontsize=10,
                               fancybox=True, shadow=True)

        # 为图例中的线条设置更粗的宽度
        for legline in legend.get_lines():
            legline.set_linewidth(3)

        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加数据统计信息
        info_text = f"Curves: {len(plot_info)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 调整布局
        plt.tight_layout()

        # 保存图片
        safe_column_name = column.strip().replace('/', '_').replace(':', '_').replace(' ', '_')
        output_path = os.path.join(output_folder, f'{safe_column_name}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  → 保存图片: {output_path}")

        # 关闭图形以释放内存
        plt.close()

    print(f"\n所有对比图已保存到: {output_folder}")


def check_data_differences(folder_path):
    """详细检查各文件数据的差异"""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    print("\n=== 数据差异详细检查 ===")

    # 读取所有文件
    dataframes = {}
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        dataframes[csv_file] = df

    # 选择一个样本列进行检查
    sample_column = None
    for col in df.columns:
        if 'mAP' in col or 'loss' in col:
            sample_column = col
            break

    if sample_column:
        print(f"\n检查列: {sample_column}")
        print("-" * 60)

        # 显示每个文件的前10个值
        for csv_file, df in dataframes.items():
            if sample_column in df.columns:
                values = pd.to_numeric(df[sample_column], errors='coerce').values[:10]
                print(f"\n{csv_file}:")
                print(f"前10个值: {values}")
                print(f"范围: [{np.nanmin(df[sample_column]):.4f}, {np.nanmax(df[sample_column]):.4f}]")


# 使用示例
if __name__ == "__main__":
    import sys

    # 从命令行参数获取文件夹路径，如果没有则使用默认值
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "compare"

    # 检查数据差异（帮助诊断）
    check_data_differences(folder_path)

    # 生成对比图
    plot_tensorboard_csvs(folder_path)
