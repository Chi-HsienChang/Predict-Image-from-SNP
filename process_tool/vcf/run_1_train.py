import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import allel

# 讀取 SNP 資料
callset = allel.read_vcf('filtered.vcf', numbers={'ALT': 1})
SNP_list = callset['variants/ALT']
SNP_list = SNP_list[0:100]

def convert_letters_to_numbers(letter_list):
    letter_to_number = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    number_list = [letter_to_number[letter[0]] for letter in letter_list]
    return number_list

SNP_list = convert_letters_to_numbers(SNP_list)
print("Integer SNP: ")
print(SNP_list)

# 生成測試資料
input_data = SNP_list 
image = Image.open("AR0809_SRR11471575.JPG")
output_matrix = np.array(image)

# 將輸入資料轉換為 PyTorch 張量
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
output_tensor = torch.tensor(output_matrix, dtype=torch.float32).unsqueeze(0)

# 獲取輸出矩陣的形狀
output_height, output_width = output_matrix.shape

print("output_matrix.shape", output_matrix.shape)
print("output_tensor", output_tensor.shape)

# 建立模型
class Generator(nn.Module):
    def __init__(self, output_height, output_width):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, output_height * output_width)
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, self.output_height, self.output_width)

model = Generator(output_height, output_width)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化最低損失
lowest_loss = float('inf')
best_image = None

# 訓練模型
num_epochs = 10

for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    generated_image = model(input_tensor)
    print("generated_image", generated_image.shape)
    print("output_tensor", output_tensor.shape)
    loss = criterion(generated_image, output_tensor)
    loss.backward()
    optimizer.step()

    # 更新最低損失和對應的圖像
    if loss.item() < lowest_loss:
        lowest_loss = loss.item()
        best_image = generated_image.squeeze().detach().numpy()

    generated_image_np = generated_image.squeeze().detach().numpy()
    plt.imshow(generated_image_np, cmap='gray')
    plt.axis('off')
    plt.savefig(f'generated_image_epoch.JPG', bbox_inches='tight', pad_inches=0)
    plt.close()

    target_image_np = output_tensor.squeeze().detach().numpy()
    plt.imshow(target_image_np, cmap='gray')
    plt.axis('off')
    plt.savefig(f'target_image_epoch.JPG', bbox_inches='tight', pad_inches=0)
    plt.close()

# 將最低損失的圖像保存
# plt.imshow(best_image, cmap='gray')
plt.imshow(best_image)
plt.axis('off')
plt.savefig('best_generated_image.JPG', bbox_inches='tight', pad_inches=0)
print("Image with lowest loss saved.")
