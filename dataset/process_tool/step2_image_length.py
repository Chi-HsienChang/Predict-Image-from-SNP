import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# def get_max_diameter(image):
#     # 將圖像轉換為灰度
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # 使用二值化找出非透明部分
#     _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

#     # 找出輪廓
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 計算最大直徑
#     max_diameter = 0
#     for contour in contours:
#         for i in range(len(contour)):
#             for j in range(i+1, len(contour)):
#                 distance = np.linalg.norm(contour[i] - contour[j])
#                 max_diameter = max(max_diameter, distance)

#     return max_diameter


# 指定圖片資料夾路徑和CSV文件路徑
image_folder_path = './raw_images/matting/long/'
# csv_file_path = './raw_images/dataset_wing_cm_angle.csv'
csv_file_path = './raw_images/RW_228_IDs.csv'

output_folder_path = './raw_images/length/long'


def resize_to_diameter(image, diameter_cm, pixelspercm, dpi):
    # 找出原始圖像的最大直徑
    # original_diameter = get_max_diameter(image)
    # original_diameter = 1
    
    # 計算像素與公分之間的轉換因子
    cm_per_pixel = diameter_cm * pixelspercm

    # 計算新的解析度
    new_dpi = dpi * cm_per_pixel

    # 使用PIL庫更改解析度
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize((image.shape[1], image.shape[0]), resample=Image.LANCZOS)
    pil_image.info['dpi'] = (new_dpi, new_dpi)
    
    return pil_image


def process_images_in_folder(folder_path, csv_path, output_folder_path, dpi=300):
    # 讀取CSV文件，獲取每個ID的直徑值
    diameter_data = pd.read_csv(csv_path).set_index('ID')

    # 遍歷指定資料夾下的每一個圖片文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # 根據圖片ID從CSV文件中獲取desired_diameter_cm
            image_id = filename.split('_')[0]
            desired_diameter_cm = diameter_data.loc[image_id, 'wingcm']
            pixelspercm = diameter_data.loc[image_id, 'pixelspercm']

            # 調整圖像大小
            resized_image = resize_to_diameter(image, desired_diameter_cm, pixelspercm, dpi)

            # 儲存新圖像到指定路徑
            output_path = os.path.join(output_folder_path, f'{image_id}_resized.JPG')
            resized_image.save(output_path)



# 調用函數處理資料夾中的圖片
process_images_in_folder(image_folder_path, csv_file_path, output_folder_path)


