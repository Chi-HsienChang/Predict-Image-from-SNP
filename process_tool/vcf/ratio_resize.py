import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, new_width):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 列出資料夾中的所有文件
    file_list = os.listdir(input_folder)
    
    for filename in file_list:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, new_width)

def resize_image(input_path, output_path, new_width):
    img = Image.open(input_path)
    original_width, original_height = img.size
    new_height = int((new_width / original_width) * original_height)
    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
    resized_img.save(output_path)

# 資料夾路徑
input_folder_path = 'train_image'   # 輸入資料夾的路徑
output_folder_path = 'train_image' # 輸出資料夾的路徑
new_width = 200                      # 新的寬度

# 呼叫函式進行資料夾中所有圖片的縮放
resize_images_in_folder(input_folder_path, output_folder_path, new_width)
