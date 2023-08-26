import os
import vcf

# 資料夾路徑
folder_path = 'all_vcf'

# 定義更嚴格的品質門檻值
quality_threshold = 95

# 儲存結果的資料夾
output_folder = 'filtered_vcfs'
os.makedirs(output_folder, exist_ok=True)

# 列出資料夾下的所有 VCF 檔案
vcf_files = [file for file in os.listdir(folder_path) if file.endswith('.vcf')]

# 開始處理每個 VCF 檔案
for vcf_file in vcf_files:
    vcf_reader = vcf.Reader(open(os.path.join(folder_path, vcf_file), 'r'))
    all_snps = list(vcf_reader)  # 原始的所有 SNP

    filtered_snps = []
    for record in all_snps:
        if (
            record.QUAL >= quality_threshold
            and record.INFO.get('DP') >= 25  # 最低讀數深度
            # and record.INFO.get('AF') >= 0.5  # 最低等位基因頻率
            # and record.INFO.get('AB') >= 0.5  # 最低等位基因平衡
            # and record.INFO.get('GQ') >= 90   # 最低基因型品質
            # and record.INFO.get('HWE') >= 0.005  # 基因型分布平衡
            # 添加更多的篩選條件，如功能注釋等
        ):
            filtered_snps.append(record)

    if filtered_snps:
        output_file_path = os.path.join(output_folder, vcf_file)
        with open(output_file_path, 'w') as output_file:
            vcf_writer = vcf.Writer(output_file, vcf_reader)
            for snp in filtered_snps:
                vcf_writer.write_record(snp)
        
        print(f"Original SNP count in '{vcf_file}': {len(all_snps)}")
        print(f"Filtered SNP count in '{vcf_file}': {len(filtered_snps)}")
    else:
        print(f"No SNP passed filtering in '{vcf_file}'")

print("All VCF files processed and filtered.")
