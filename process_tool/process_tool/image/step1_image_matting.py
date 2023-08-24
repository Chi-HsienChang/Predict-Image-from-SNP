import os
from rembg import remove

input_folder = '../raw_images/long'  # 輸入資料夾的路徑
output_folder = '../raw_images/matting/long'  # 輸出資料夾的路徑

# 確保輸出資料夾存在
# os.makedirs(output_folder, exist_ok=True)

# 遍歷輸入資料夾中的每個文件
for filename in os.listdir(input_folder):
    if filename.endswith('.JPG'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.JPG', '_MASK.JPG').replace('.jpg', '_MASK.JPG'))
        
        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input_data = i.read()
                output_data = remove(input_data)
                o.write(output_data)
