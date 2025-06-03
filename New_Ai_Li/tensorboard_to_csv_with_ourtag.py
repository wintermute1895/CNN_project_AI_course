#!/usr/bin/env python3
import argparse
from tbparse import SummaryReader
import os
import re


def sanitize_for_filename(text):
    """清理文本用于文件名"""
    # 保留&符号，但替换其他特殊字符
    return re.sub(r'[<>:"|?*\\/]', '_', text)


def export_tensorboard_with_prefix(log_dir, output_dir, prefix,
                                   keep_ampersand=True, create_subdir=False):
    """导出TensorBoard数据并添加前缀"""

    # 处理前缀
    if not keep_ampersand:
        prefix = prefix.replace('&', '_')

    # 创建输出目录
    if create_subdir:
        output_dir = os.path.join(output_dir, sanitize_for_filename(prefix))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Configuration:")
    print(f"  - Log directory: {log_dir}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Prefix: {prefix}")
    print()

    # 读取数据
    reader = SummaryReader(log_dir)
    scalars_df = reader.scalars

    if scalars_df.empty:
        print("No scalar data found!")
        return

    # 导出统计
    exported_files = []
    total_records = 0

    # 按标签导出
    unique_tags = scalars_df['tag'].unique()
    print(f"Exporting {len(unique_tags)} metrics...\n")

    for i, tag in enumerate(unique_tags, 1):
        tag_data = scalars_df[scalars_df['tag'] == tag]

        # 构建文件名
        safe_tag = sanitize_for_filename(tag)
        filename = f"{prefix}_{safe_tag}.csv"
        output_path = os.path.join(output_dir, filename)

        # 保存
        tag_data.to_csv(output_path, index=False)

        # 记录信息
        exported_files.append(filename)
        total_records += len(tag_data)

        # 显示进度
        print(f"[{i:3d}/{len(unique_tags)}] {filename}")
        print(f"         → {len(tag_data)} records")

    # 打印摘要
    print(f"\n{'=' * 60}")
    print(f"Export completed!")
    print(f"  - Files created: {len(exported_files)}")
    print(f"  - Total records: {total_records:,}")
    print(f"  - Output directory: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description='Export TensorBoard logs with custom prefix'
    )
    parser.add_argument('logdir', help='TensorBoard log directory')
    parser.add_argument('-o', '--output', default='exported_csvs',
                        help='Output directory (default: exported_csvs)')
    parser.add_argument('-p', '--prefix', required=True,
                        help='Prefix for output files (e.g., "our_model&epoch=35")')
    parser.add_argument('--no-ampersand', action='store_true',
                        help='Replace & with _ in filenames')
    parser.add_argument('--subdir', action='store_true',
                        help='Create subdirectory with prefix name')

    args = parser.parse_args()

    export_tensorboard_with_prefix(
        args.logdir,
        args.output,
        args.prefix,
        keep_ampersand=not args.no_ampersand,
        create_subdir=args.subdir
    )


if __name__ == "__main__":
    main()


#python tensorboard_to_csv_with_ourtag.py yolov5/runs/train/exp7 -o se_backbone_r32_csvs -p "se_backbone_r32&epoch=99"