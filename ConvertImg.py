import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

def csv_to_grayscale_image(task):
    csv_file, output_file = task

    # 读取CSV文件
    data = pd.read_csv(csv_file, header=None)

    # 检查数据是否为30x30矩阵
    if data.shape != (30, 30):
        print(f"File {csv_file} does not contain a 30x30 matrix.")
        return csv_file, False

    # 将数据转换为numpy数组并缩放到0-255
    matrix = data.to_numpy()
    scaled_matrix = (matrix * 255).astype(np.uint8)

    # 使用Pillow生成灰度图像
    image = Image.fromarray(scaled_matrix, mode='L')

    # 保存图像
    image.save(output_file)
    return csv_file, True

def process_folder(input_folder, output_folder):
    tasks = []

    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)

                # 创建对应的输出文件夹结构
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 构建输出文件路径
                base_filename = os.path.basename(csv_file)
                filename_without_ext = os.path.splitext(base_filename)[0]
                output_file = os.path.join(output_dir, f"{filename_without_ext}.png")

                # 将任务添加到列表中
                tasks.append((csv_file, output_file))

    # 使用并行处理来处理文件
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(csv_to_grayscale_image, task) for task in tasks]

        for future in as_completed(futures):
            csv_file, success = future.result()
            if success:
                print(f"Processed {csv_file}")
            else:
                print(f"Failed to process {csv_file}")

# 指定输入文件夹和输出文件夹
input_folder = '/home/Z/UANETTest_Multi/10-30-probability--500n2'
output_folder = '/home/Z/UANETTest_Multi/10-30-probability--500n2-Img--30*30'

# 处理文件夹
process_folder(input_folder, output_folder)
