from PIL import Image

# 開啟圖片
image_path = "CA0519_length.JPG"
img = Image.open(image_path)

# 平移距離（像素）
shift_x = 300
shift_y = 60

# 定義仿射變換矩陣（平移）
transform_matrix = (1, 0, shift_x, 0, 1, shift_y)

# 進行仿射變換（保持畫質和大小）
shifted_img = img.transform(img.size, Image.AFFINE, transform_matrix, resample=Image.NONE)

# 儲存平移後的圖片
shifted_img.save("平移後的圖片.jpg")
# shifted_img.save(image_path)

print("已完成圖片平移，保持原始畫質和大小")
