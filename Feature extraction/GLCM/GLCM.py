import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import spectral.io.envi as envi

def scale_and_quantize_image(image, levels):
    """
    将图像数据缩放到指定灰度级并量化。
    """
    if image.min() == image.max():
        quantized_image = np.zeros_like(image, dtype=np.uint8)
    else:
        scaled_image = np.interp(image, (image.min(), image.max()), (0, levels-1))
        quantized_image = scaled_image.astype(np.uint8)
    return quantized_image

def process_band_data(band_data, levels=16):
    """
    处理单个波段的数据，提取四个角度的 GLCM 特征。
    """
    quantized_data = scale_and_quantize_image(band_data, levels)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
    results = {}
    for idx, angle in enumerate(angles):
        glcm = graycomatrix(quantized_data, [1], [angle], levels=levels, symmetric=True, normed=True)
        results[f'对比度_角度{idx}'] = graycoprops(glcm, 'contrast')[0, 0]
        results[f'相关性_角度{idx}'] = graycoprops(glcm, 'correlation')[0, 0]
        results[f'能量_角度{idx}'] = graycoprops(glcm, 'energy')[0, 0]
        results[f'逆差矩_角度{idx}'] = graycoprops(glcm, 'ASM')[0, 0]
    return results

def process_folder(folder_path, bands, levels):
    """
    处理文件夹中的 HDR 文件，提取所有波段的 GLCM 特征。
    """
    hdr_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.hdr')]
    if len(hdr_files) != 1:
        print(f'文件夹 {folder_path} 中未找到唯一的hdr文件')
        return None
    hdr_file_path = os.path.join(folder_path, hdr_files[0])
    matcha_hyp_data = envi.open(hdr_file_path)
    if matcha_hyp_data.shape[2] != bands:
        print(f'波段数不匹配，预期{bands}，但实际为{matcha_hyp_data.shape[2]}')
        return None
    all_results = []
    for i in range(bands):
        band_data = matcha_hyp_data.read_band(i)
        band_results = process_band_data(band_data, levels)
        band_results['波段'] = i  # 添加波段编号
        all_results.append(band_results)
    return all_results

def main(data_folder, output_folder, bands, levels):
    """
    主函数，遍历数据文件夹，处理每个文件夹并保存结果。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            try:
                folder_num = int(folder_name)
                print(f'处理文件夹：{folder_name}')
                results = process_folder(folder_path, bands, levels)
                if results:
                    df = pd.DataFrame(results)
                    output_file = os.path.join(output_folder, f'{folder_num + 20:03d}.xlsx')# 保存文件名，对应文件名
                    df.to_excel(output_file, index=False)
                    print(f'特征值已保存到 {output_file}')
            except ValueError:
                print(f'文件夹名 {folder_name} 不是有效的数字，跳过处理。')
    print('处理完成。')

# 参数设置
data_folder = r"D:\红茶数据2024.0423\多光谱\萎凋过程-多光谱\红外萎凋6h（21-120）\3h（21-40）\3_processed" # 输入数据文件夹路径
output_folder = r'D:\红茶数据2024.0423\多光谱\萎凋过程-多光谱\结果\红外6小时\GLCM'  # 输出文件夹路径
bands = 25  # 波段数
levels = 16  # 灰度级

# 运行主函数
main(data_folder, output_folder, bands, levels)