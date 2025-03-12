import os
import pandas as pd

# 指定文件夹路径和输出路径
folder_path = r"D:\红茶数据2024.0423\多光谱\萎凋过程-多光谱\结果\红外6小时\GLCM"
output_path = os.path.join(folder_path, 'GLCM.xlsx')

# 列出所有.xlsx文件，并按文件名数字排序
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and f[:-5].isdigit()]
files.sort(key=lambda x: int(x[:-5]))

# 创建一个空的数据框来存储所有平均值
all_avgs = pd.DataFrame()

for f in files:
    file_path = os.path.join(folder_path, f)
    try:
        # 读取Excel文件，默认读取第一个sheet
        df = pd.read_excel(file_path)
        # 提取2到26行的数据，假设行索引从0开始
        data = df.iloc[1:26, :]
        # 计算平均值
        avg = data.mean()
        # 将avg转换为DataFrame并转置，然后concat到all_avgs
        avg_df = avg.to_frame().T
        all_avgs = pd.concat([all_avgs, avg_df], ignore_index=True)
    except Exception as e:
        print(f'处理文件{f}时出错: {e}')

# 保存所有平均值到新的Excel文件
all_avgs.to_excel(output_path, index=False)