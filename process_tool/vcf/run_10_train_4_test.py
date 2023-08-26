import os
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

# 分割訓練集和測試集
# split_ratio = 0.8  # 80% 用於訓練，20% 用於測試
# split_index = int(len(SNP_list) * split_ratio)
train_input_data = [SNP_list]*10

test_input_data = [SNP_list]*4
test_input_tensor = torch.tensor(test_input_data, dtype=torch.float32).unsqueeze(0)

# 讀取訓練圖像
train_images = []
train_path = 'train_image'
for filename in os.listdir(train_path):
    if filename.endswith(".JPG"):
        image_path = os.path.join(train_path, filename)
        image = Image.open(image_path)
        train_images.append(np.array(image))

train_output_data = torch.tensor(train_images, dtype=torch.float32)

# 讀取測試圖像
test_images = []
test_path = 'test_image'
for filename in os.listdir(test_path):
    if filename.endswith(".JPG"):
        image_path = os.path.join(test_path, filename)
        image = Image.open(image_path)
        test_images.append(np.array(image))

test_output_tensor = torch.tensor(test_images, dtype=torch.float32)

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

output_height, output_width = test_output_tensor.shape[1], test_output_tensor.shape[2]
model = Generator(output_height, output_width)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 100
train_input_tensor = torch.tensor(train_input_data, dtype=torch.float32).unsqueeze(0)
train_output_tensor = torch.tensor(train_output_data, dtype=torch.float32).unsqueeze(0)

for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    generated_images = model(train_input_tensor)
    loss = criterion(generated_images, train_output_tensor)
    loss.backward()
    optimizer.step()

# 測試模型
with torch.no_grad():
    test_generated_images = model(test_input_tensor)
    test_loss = criterion(test_generated_images, test_output_tensor)

# 顯示測試結果
for i in range(len(test_generated_images)):
    plt.imshow(test_generated_images[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(f'test_generated_image_{i}.JPG', bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(test_output_tensor[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(f'test_target_image_{i}.JPG', bbox_inches='tight', pad_inches=0)
    plt.close()

print("Test completed.")
