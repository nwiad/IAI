import json
import re
from utils import filter_by_table, extract, extract_triplets

fin = "usual_train_new.txt"
fout = "smp.txt"
tables_dir = "../../tables/"

with open(fout, "w", encoding="utf-8") as file_out:
    with open(fin, "r", encoding="gbk") as file_in:
        while line := file_in.readline():
            filtered_content = re.sub("[^\u4e00-\u9fa5]+", "", line)
            file_out.write(filtered_content+"\n")
print("done filtering by Chinese")

filter_by_table(fin="smp.txt", fout="filtered_smp.txt")
print("done filtering by table")

sing_table = {}
dual_table = {}
extract("filtered_smp.txt", sing_table=sing_table, dual_table=dual_table)
print("done extracting characters and dual-words")
with open(tables_dir+"sing-smp.txt", "w", encoding="utf-8") as file_out:
    file_out.write(json.dumps(sing_table, ensure_ascii=False))
    file_out.close()
with open(tables_dir+"dual-smp.txt", "w", encoding="utf-8") as file_out:
    file_out.write(json.dumps(dual_table, ensure_ascii=False))
    file_out.close()
print("done building sing_table and dual_table of smp")

tri_table = {}
extract_triplets("filtered_smp.txt", tri_table)
print("done extracting triplets")
with open(tables_dir+"tri-smp.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(tri_table, ensure_ascii=False))
print("done building tri_table of smp")