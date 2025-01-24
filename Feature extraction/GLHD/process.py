import os
import pandas as pd

# 指定文件夹路径和输出路径
folder_path = r'D:\茶叶干燥过程\茶叶多光谱图像\热风第批140个样+水分\GLHD'
output_path = os.path.join(folder_path, 'GLHD.xlsx')

# 列出所有.xlsx文件，并按文件名数字排序
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and f[:-5].isdigit()]
files.sort(key=lambda x: int(x[:-5]))

# 创建一个空的数据框来存储所有平均值
all_avgs = pd.DataFrame()

for f in files:
    file_path = os.path.join(folder_path, f)
    try:
        # 读取Excel文件，默认读取第一个sheet
        df = pd.read_excel(file_path, header=None)  # 假设没有表头
        # 提取1到25行的数据（行索引从0开始，对应1-25行）
        data = df.iloc[0:25, :]  # 提取前25行
        # 计算平均值
        avg = data.mean()
        # 将avg转换为DataFrame并转置，然后concat到all_avgs
        avg_df = avg.to_frame().T
        all_avgs = pd.concat([all_avgs, avg_df], ignore_index=True)
    except Exception as e:
        print(f'处理文件{f}时出错: {e}')

# 保存所有平均值到新的Excel文件
all_avgs.to_excel(output_path, index=False)
print(f'所有文件的平均值已保存到: {output_path}')