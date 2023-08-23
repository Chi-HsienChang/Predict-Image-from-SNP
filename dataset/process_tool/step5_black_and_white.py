import os
from PIL import Image

# 資料夾路徑
folder_path = "./short"

# 獲取資料夾中的所有圖片
image_files = [f for f in os.listdir(folder_path) if f.endswith('.JPG')]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    
    # 開啟原始 RGB 圖片
    rgb_img = Image.open(image_path)
    
    # 將 RGB 圖片轉換為灰階圖片
    gray_img = rgb_img.convert("L")
    
    # 儲存灰階圖片（輸出的檔名加上 "_gray" 後綴）
    gray_img.save(os.path.join(folder_path, f"{os.path.splitext(image_file)[0]}_gray.JPG"))

print("已完成將三維 RGB 圖片轉換為一維灰階圖片")
