import os
import vcf

# 指定包含VCF文件的資料夾
vcf_folder = 'common_SNP_vcfs'  # 請替換為您的VCF資料夾路徑

# 讀取每個 VCF 文件，印出碱基序列和染色體位置
for filename in os.listdir(vcf_folder):
    if filename.endswith(".vcf"):
        vcf_path = os.path.join(vcf_folder, filename)
        vcf_reader = vcf.Reader(filename=vcf_path)
        
        for record in vcf_reader:
            chromosome = record.CHROM
            position = record.POS
            bases = record.REF
            print(f"VCF File: {filename}, Chromosome: {chromosome}, Position: {position}, Bases: {bases}")
        print("--------------------------------")



import os
import vcf

# 指定包含VCF文件的資料夾
vcf_folder = 'common_SNP_vcfs'  # 請替換為您的VCF資料夾路徑

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

print(len(int_values_list))







