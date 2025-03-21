import os
from PIL import Image

# 设置主文件夹路径
main_folder = r"D:\红茶数据2024.0423\多光谱\萎凋过程-多光谱\红外萎凋6h（21-120）\15h（101-120）\15_processed"


# 遍历1到20的文件夹
for folder_number in range(1, 21):
    # 构建子文件夹路径
    folder_path = os.path.join(main_folder, str(folder_number))

    # 构建interestingspace文件夹路径
    interestingspace_folder = os.path.join(folder_path, '25个波段对应的图像', 'interestingspace')
    os.makedirs(interestingspace_folder, exist_ok=True)

    # 遍历25个波段
    for i in range(1, 26):
        # 构建文件路径
        filename = os.path.join(folder_path, '25个波段对应的图像', f'{i}.bmp')

        # 读取图像
        A = Image.open(filename)

        # 裁剪图像
        B = A.crop((110, 50, 250, 180))  
        # 修改裁剪的区域，参数为 (left, upper, right, lower)，向下和向右加值为正，向上和向左为负

        # 构建保存文件路径
        save_path = os.path.join(interestingspace_folder, f'{i}.bmp')

        # 保存裁剪后的图像
        B.save(save_path)

        print(f'处理文件夹 {folder_number} 中的波段 {i}，保存到 {save_path}')

print('处理完成。')