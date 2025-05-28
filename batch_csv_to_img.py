#用法python batch_csv_to_img.py file1 file2 “”
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from glob import glob

def csv_to_plot(csv_path, output_path=None):
    data = pd.read_csv(csv_path, sep=None, engine='python')

    required_cols = ['Step', 'Value']
    for col in required_cols:
        if col not in data.columns:
            print(f"文件{csv_path}缺少列: {col}，跳过。")
            return

    plt.figure(figsize=(10,6))
    plt.plot(data['Step'], data['Value'], label='Value')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(os.path.basename(csv_path))
    plt.legend()
    plt.grid(True)

    if output_path is None:
        output_path = os.path.splitext(csv_path)[0] + '.png'

    plt.savefig(output_path)
    plt.close()
    print(f"已保存: {output_path}")

def batch_convert(folder_path, output_folder=None):
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        print("没有找到csv文件")
        return

    for csv_file in csv_files:
        if output_folder:
            output_path = os.path.join(output_folder, os.path.basename(csv_file).replace('.csv', '.png'))
        else:
            output_path = None
        csv_to_plot(csv_file, output_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python batch_csv_to_img.py folder_path [output_folder]")
        sys.exit(1)
    folder_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    batch_convert(folder_path, output_folder)

