import cv2
import numpy as np
import os

def find_max_diameter(image_path):
    # 讀取圖片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 將圖片轉換為二值圖像
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_diameter = 0
    
    for contour in contours:
        # 計算輪廓的最小外接圓
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter = int(radius * 2)
        if diameter > max_diameter:
            max_diameter = diameter
    
    return max_diameter

# 資料夾路徑
folder_path = "./after_length/short"

# 獲取資料夾中的所有圖片
image_files = [f for f in os.listdir(folder_path) if f.endswith('.JPG')]

data = []

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    max_diameter = find_max_diameter(image_path)
    data.append(f"{image_file}_{max_diameter}_px")

# 將數據分割並寫入 CSV 文件
csv_path = "max_diameter_short.csv"
with open(csv_path, 'w') as f:
    for item in data:
        parts = item.split('_')
        csv_line = ','.join(parts)  # 使用逗號分隔各部分
        f.write("%s\n" % csv_line)

print(f"已將數據保存到 {csv_path}")


# for image_file in image_files:
#     image_path = os.path.join(folder_path, image_file)
#     max_diameter = find_max_diameter(image_path)
#     print(f"{image_file}_{max_diameter}_px")
