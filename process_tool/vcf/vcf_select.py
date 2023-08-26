import os
import vcf

# 指定包含VCF文件的資料夾
vcf_folder = 'filtered_vcfs'  # 請替換為您的VCF資料夾路徑

# 創建一個字典來存儲每個VCF文件的SNP位置
vcf_snp_positions = {}

# 讀取每個VCF文件，識別SNP位置
for filename in os.listdir(vcf_folder):
    if filename.endswith(".vcf"):
        vcf_path = os.path.join(vcf_folder, filename)
        vcf_reader = vcf.Reader(filename=vcf_path)
        snp_positions = set()
        for record in vcf_reader:
            # 篩選掉base長度大於1的條件
            if len(record.REF) == 1 and all(len(alt) == 1 for alt in record.ALT):
                snp_positions.add((record.CHROM, record.POS))
        vcf_snp_positions[filename] = snp_positions

# 找到所有VCF文件共同的SNP位置
common_snp_positions = set.intersection(*vcf_snp_positions.values())

print(len(common_snp_positions))

# 過濾每個VCF文件，只保留共同的SNP位置，並生成對應數量的新VCF文件
for filename, snp_positions in vcf_snp_positions.items():
    output_vcf_path = os.path.join(vcf_folder, f"filtered_{filename}")
    vcf_reader = vcf.Reader(filename=os.path.join(vcf_folder, filename))
    output_vcf = vcf.Writer(open(output_vcf_path, 'w'), vcf_reader)
    for record in vcf_reader:
        if (record.CHROM, record.POS) in common_snp_positions:
            output_vcf.write_record(record)
    output_vcf.close()

print("已生成對應數量的過濾後VCF文件。")












# import os
# import vcf

# # 指定包含VCF文件的資料夾
# vcf_folder = 'filtered_vcfs'  # 請替換為您的VCF資料夾路徑

# # 創建一個字典來存儲每個VCF文件的SNP位置
# vcf_snp_positions = {}

# # 讀取每個VCF文件，識別SNP位置
# for filename in os.listdir(vcf_folder):
#     if filename.endswith(".vcf"):
#         vcf_path = os.path.join(vcf_folder, filename)
#         vcf_reader = vcf.Reader(filename=vcf_path)
#         snp_positions = set()
#         for record in vcf_reader:
#             snp_positions.add((record.CHROM, record.POS))
#         vcf_snp_positions[filename] = snp_positions

# # 找到所有VCF文件共同的SNP位置
# common_snp_positions = set.intersection(*vcf_snp_positions.values())

# print(len(common_snp_positions))

# # 過濾每個VCF文件，只保留共同的SNP位置，並生成對應數量的新VCF文件
# for filename, snp_positions in vcf_snp_positions.items():
#     output_vcf_path = os.path.join(vcf_folder, f"filtered_{filename}")
#     vcf_reader = vcf.Reader(filename=os.path.join(vcf_folder, filename))
#     output_vcf = vcf.Writer(open(output_vcf_path, 'w'), vcf_reader)
#     for record in vcf_reader:
#         if (record.CHROM, record.POS) in common_snp_positions:
#             output_vcf.write_record(record)
#     output_vcf.close()

# print("已生成對應數量的過濾後VCF文件。")
