from tbparse import SummaryReader
import os
import re
import platform
def sanitize_filename(filename):
    """清理文件名，移除或替换不允许的字符"""

    # 根据操作系统选择不允许的字符
    if platform.system() == 'Windows':
        # Windows 不允许的字符
        invalid_chars = r'[<>:"|?*\\/]'
    else:
        # Unix/Linux 主要是斜杠
        invalid_chars = r'[/]'

    # 替换不允许的字符为下划线
    safe_name = re.sub(invalid_chars, '_', filename)

    # 移除文件名末尾的点和空格（Windows 特殊要求）
    safe_name = safe_name.rstrip('. ')

    # 确保文件名不为空
    if not safe_name:
        safe_name = 'unnamed'

    return safe_name


def export_by_tags(log_dir, output_dir='exported_csvs'):
    """按标签分别导出数据，支持跨平台"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 读取数据
        print(f"Reading TensorBoard logs from: {log_dir}")
        reader = SummaryReader(log_dir)
        scalars_df = reader.scalars

        if scalars_df.empty:
            print("Warning: No scalar data found in the logs!")
            return

        # 获取所有唯一的标签
        unique_tags = scalars_df['tag'].unique()
        print(f"\nFound {len(unique_tags)} unique tags:")

        # 记录导出结果
        success_count = 0
        failed_tags = []

        # 按标签导出
        for i, tag in enumerate(unique_tags, 1):
            try:
                # 过滤特定标签的数据
                tag_data = scalars_df[scalars_df['tag'] == tag]

                # 清理文件名
                safe_filename = sanitize_filename(tag)
                output_path = os.path.join(output_dir, f"{safe_filename}.csv")

                # 如果文件名已存在，添加数字后缀
                base_path = output_path
                counter = 1
                while os.path.exists(output_path):
                    output_path = os.path.join(output_dir, f"{safe_filename}_{counter}.csv")
                    counter += 1

                # 保存CSV
                tag_data.to_csv(output_path, index=False)

                # 验证保存结果
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"  [{i}/{len(unique_tags)}] ✓ {tag}")
                    print(f"        → {output_path} ({len(tag_data)} records)")
                    success_count += 1
                else:
                    raise Exception("File created but is empty")

            except Exception as e:
                print(f"  [{i}/{len(unique_tags)}] ✗ {tag}")
                print(f"        Error: {str(e)}")
                failed_tags.append(tag)

        # 打印摘要
        print(f"\n{'=' * 50}")
        print(f"Export Summary:")
        print(f"  - Total tags: {len(unique_tags)}")
        print(f"  - Successfully exported: {success_count}")
        print(f"  - Failed: {len(failed_tags)}")

        if failed_tags:
            print(f"\nFailed tags:")
            for tag in failed_tags:
                print(f"  - {tag}")

    except Exception as e:
        print(f"\nError reading TensorBoard logs: {str(e)}")
        print("Please check if the log directory is correct and contains valid TensorBoard files.")


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际路径
    log_directory = 'yolov5/runs/train/exp2'
    export_by_tags(log_directory, 'exported_csvs')
