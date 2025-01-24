import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import spectral.io.envi as envi
import pywt
from scipy import stats

def scale_and_quantize_image(image, levels):
    if image.min() == image.max():
        quantized_image = np.zeros_like(image, dtype=np.uint8)
    else:
        scaled_image = np.interp(image, (image.min(), image.max()), (0, levels-1))
        quantized_image = scaled_image.astype(np.uint8)
    return quantized_image

def process_wavelet_features(band_data, wavelet='db4', level=2):
    """
    提取小波变换特征
    :param band_data: 输入波段数据
    :param wavelet: 小波基函数，默认使用'db4'
    :param level: 小波分解层数，默认2层
    :return: 小波变换特征（均值、方差、偏度、峰度）
    """
    coeffs = pywt.wavedec2(band_data, wavelet, level=level)
    approx = coeffs[0]  # 近似分量
    details_level2 = coeffs[1]  # 第二层细节分量
    details_level1 = coeffs[2]  # 第一层细节分量

    # 提取近似分量的统计特征
    approx_mean = np.mean(approx)
    approx_var = np.var(approx)
    approx_skew = stats.skew(approx.flatten())
    approx_kurt = stats.kurtosis(approx.flatten())

    # 提取第二层细节分量的统计特征（水平、垂直、对角线）
    h_mean_l2, h_var_l2 = np.mean(details_level2[0]), np.var(details_level2[0])
    h_skew_l2, h_kurt_l2 = stats.skew(details_level2[0].flatten()), stats.kurtosis(details_level2[0].flatten())
    
    v_mean_l2, v_var_l2 = np.mean(details_level2[1]), np.var(details_level2[1])
    v_skew_l2, v_kurt_l2 = stats.skew(details_level2[1].flatten()), stats.kurtosis(details_level2[1].flatten())
    
    d_mean_l2, d_var_l2 = np.mean(details_level2[2]), np.var(details_level2[2])
    d_skew_l2, d_kurt_l2 = stats.skew(details_level2[2].flatten()), stats.kurtosis(details_level2[2].flatten())

    # 提取第一层细节分量的统计特征（水平、垂直、对角线）
    h_mean_l1, h_var_l1 = np.mean(details_level1[0]), np.var(details_level1[0])
    h_skew_l1, h_kurt_l1 = stats.skew(details_level1[0].flatten()), stats.kurtosis(details_level1[0].flatten())
    
    v_mean_l1, v_var_l1 = np.mean(details_level1[1]), np.var(details_level1[1])
    v_skew_l1, v_kurt_l1 = stats.skew(details_level1[1].flatten()), stats.kurtosis(details_level1[1].flatten())
    
    d_mean_l1, d_var_l1 = np.mean(details_level1[2]), np.var(details_level1[2])
    d_skew_l1, d_kurt_l1 = stats.skew(details_level1[2].flatten()), stats.kurtosis(details_level1[2].flatten())

    return (approx_mean, approx_var, approx_skew, approx_kurt,
            h_mean_l2, h_var_l2, h_skew_l2, h_kurt_l2,
            v_mean_l2, v_var_l2, v_skew_l2, v_kurt_l2,
            d_mean_l2, d_var_l2, d_skew_l2, d_kurt_l2,
            h_mean_l1, h_var_l1, h_skew_l1, h_kurt_l1,
            v_mean_l1, v_var_l1, v_skew_l1, v_kurt_l1,
            d_mean_l1, d_var_l1, d_skew_l1, d_kurt_l1)

def process_band_data(band_data, levels=16):
    """
    处理单波段数据，提取GLCM和小波特征
    :param band_data: 输入波段数据
    :param levels: 灰度级数
    :return: 包含GLCM和小波特性的字典
    """
    # GLCM特征
    quantized_data = scale_and_quantize_image(band_data, levels)
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_features = {'contrast': [], 'correlation': [], 'ASM': [], 'homogeneity': [], 'dissimilarity': []}
    for angle in angles:
        for distance in distances:
            glcm = graycomatrix(quantized_data, distances=[distance], angles=[angle], levels=levels, symmetric=True, normed=True)
            glcm_features['contrast'].append(graycoprops(glcm, 'contrast')[0, 0])
            glcm_features['correlation'].append(graycoprops(glcm, 'correlation')[0, 0])
            glcm_features['ASM'].append(graycoprops(glcm, 'ASM')[0, 0])
            glcm_features['homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
            glcm_features['dissimilarity'].append(graycoprops(glcm, 'dissimilarity')[0, 0])
    mean_contrast = np.mean(glcm_features['contrast'])
    mean_correlation = np.mean(glcm_features['correlation'])
    mean_ASM = np.mean(glcm_features['ASM'])
    mean_homogeneity = np.mean(glcm_features['homogeneity'])
    mean_dissimilarity = np.mean(glcm_features['dissimilarity'])

    # 小波特征
    wavelet_features = process_wavelet_features(band_data)

    # 合并GLCM和小波特征
    features_dict = {
        '对比度 (Contrast)': mean_contrast,
        '相关性 (Correlation)': mean_correlation,
        'ASM': mean_ASM,
        '同质性 (Homogeneity)': mean_homogeneity,
        '不相似性 (Dissimilarity)': mean_dissimilarity,
        '近似均值 (Approx_Mean)': wavelet_features[0],
        '近似方差 (Approx_Var)': wavelet_features[1],
        '近似偏度 (Approx_Skew)': wavelet_features[2],
        '近似峰度 (Approx_Kurt)': wavelet_features[3],
        '水平均值_L2 (H_Mean_L2)': wavelet_features[4],
        '水平方差_L2 (H_Var_L2)': wavelet_features[5],
        '水平偏度_L2 (H_Skew_L2)': wavelet_features[6],
        '水平峰度_L2 (H_Kurt_L2)': wavelet_features[7],
        '垂直均值_L2 (V_Mean_L2)': wavelet_features[8],
        '垂直方差_L2 (V_Var_L2)': wavelet_features[9],
        '垂直偏度_L2 (V_Skew_L2)': wavelet_features[10],
        '垂直峰度_L2 (V_Kurt_L2)': wavelet_features[11],
        '对角线均值_L2 (D_Mean_L2)': wavelet_features[12],
        '对角线方差_L2 (D_Var_L2)': wavelet_features[13],
        '对角线偏度_L2 (D_Skew_L2)': wavelet_features[14],
        '对角线峰度_L2 (D_Kurt_L2)': wavelet_features[15],
        '水平均值_L1 (H_Mean_L1)': wavelet_features[16],
        '水平方差_L1 (H_Var_L1)': wavelet_features[17],
        '水平偏度_L1 (H_Skew_L1)': wavelet_features[18],
        '水平峰度_L1 (H_Kurt_L1)': wavelet_features[19],
        '垂直均值_L1 (V_Mean_L1)': wavelet_features[20],
        '垂直方差_L1 (V_Var_L1)': wavelet_features[21],
        '垂直偏度_L1 (V_Skew_L1)': wavelet_features[22],
        '垂直峰度_L1 (V_Kurt_L1)': wavelet_features[23],
        '对角线均值_L1 (D_Mean_L1)': wavelet_features[24],
        '对角线方差_L1 (D_Var_L1)': wavelet_features[25],
        '对角线偏度_L1 (D_Skew_L1)': wavelet_features[26],
        '对角线峰度_L1 (D_Kurt_L1)': wavelet_features[27]
    }
    return features_dict

def process_folder(folder_path, bands, levels):
    """
    处理文件夹中的多光谱数据
    :param folder_path: 文件夹路径
    :param bands: 波段数
    :param levels: 灰度级数
    :return: 包含所有波段特征的DataFrame
    """
    hdr_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.hdr')]
    if len(hdr_files) != 1:
        print(f'文件夹 {folder_path} 中未找到唯一的hdr文件')
        return None
    hdr_file_path = os.path.join(folder_path, hdr_files[0])
    matcha_hyp_data = envi.open(hdr_file_path)
    
    features_list = []
    for i in range(bands):
        band_data = matcha_hyp_data.read_band(i)
        features = process_band_data(band_data, levels)
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    return df

def main(data_folder, output_folder, bands, levels):
    """
    主函数，处理所有文件夹中的数据
    :param data_folder: 数据文件夹路径
    :param output_folder: 输出文件夹路径
    :param bands: 波段数
    :param levels: 灰度级数
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            try:
                folder_num = int(folder_name)
                print(f'处理文件夹：{folder_name}')
                df = process_folder(folder_path, bands, levels)
                if df is not None:
                    output_file = os.path.join(output_folder, f'{folder_num + 100:03d}.xlsx')
                    df.to_excel(output_file, index=False)
                    print(f'特征值已保存到 {output_file}')
            except ValueError:
                print(f'文件夹名 {folder_name} 不是有效的数字，跳过处理。')
    print('处理完成。')

# 参数设置
data_folder = r'D:\茶叶干燥过程\茶叶多光谱图像\热风第二批140个样+水分\6足火后_processed'
output_folder = r'D:\茶叶干燥过程\茶叶多光谱图像\热风第二批140个样+水分\纹理5'
bands = 25
levels = 16  # 设置16个灰度级

main(data_folder, output_folder, bands, levels)