import os
import vcf
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import allel



# 指定包含VCF文件的資料夾
input_dim = 261
vcf_folder = 'train_vcf'  # 請替換為您的VCF資料夾路徑

# 定義碱基值對應的整數字典
base_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

# 創建一個列表來存儲轉換後的整數值
int_values_list = []

# 讀取每個 VCF 文件，將碱基值轉換成整數並存儲
for filename in os.listdir(vcf_folder):
    if filename.endswith(".vcf"):
        vcf_path = os.path.join(vcf_folder, filename)
        vcf_reader = vcf.Reader(filename=vcf_path)
        int_values = []
        for record in vcf_reader:
            for base in record.REF:
                int_value = base_to_int.get(base.upper(), -1)  # 若不是 A/T/C/G，返回 -1
                int_values.append(int_value)
        print(len(int_values))
        int_values_list.append(int_values)

train_input_data = int_values_list

vcf_folder = 'test_vcf'  # 請替換為您的VCF資料夾路徑

# 定義碱基值對應的整數字典
base_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

# 創建一個列表來存儲轉換後的整數值
int_values_list = []

# 讀取每個 VCF 文件，將碱基值轉換成整數並存儲
for filename in os.listdir(vcf_folder):
    if filename.endswith(".vcf"):
        vcf_path = os.path.join(vcf_folder, filename)
        vcf_reader = vcf.Reader(filename=vcf_path)
        int_values = []
        for record in vcf_reader:
            for base in record.REF:
                int_value = base_to_int.get(base.upper(), -1)  # 若不是 A/T/C/G，返回 -1
                int_values.append(int_value)
        print(len(int_values))
        int_values_list.append(int_values)


test_input_data = int_values_list

different_positions = 0
position_count = len(test_input_data[0])  # 假設每組列表的長度相同

for i in range(position_count):
    values_at_position = set(item[i] for item in test_input_data)
    if len(values_at_position) > 1:
        different_positions += 1

print(f"有 {different_positions} 個位置的值不同。")

# print("test_input_data", test_input_data)

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

test_output_data = torch.tensor(test_images, dtype=torch.float32)

train_input_tensor = torch.tensor(train_input_data, dtype=torch.float32)

# train_output_tensor = torch.tensor(train_output_data, dtype=torch.float32)
train_output_tensor = train_output_data.clone().detach()

test_input_tensor = torch.tensor(test_input_data, dtype=torch.float32)

# test_output_tensor = torch.tensor(test_output_data, dtype=torch.float32)
test_output_tensor = test_output_data.clone().detach()


# 建立模型
class Generator(nn.Module):
    def __init__(self, output_height, output_width):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
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
