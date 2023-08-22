from PIL import Image

# 開啟圖片
image_path = "CA0519_length.JPG"
img = Image.open(image_path)

# 旋轉角度（逆時針方向）
angle = -7

# 旋轉圖片（不改變畫質）
rotated_img = img.rotate(angle, resample=Image.NONE, expand=False)

# 儲存旋轉後的圖片
rotated_img.save("旋轉後的圖片.jpg")
# rotated_img.save(image_path)

print("已完成圖片旋轉，保持原始畫質")

