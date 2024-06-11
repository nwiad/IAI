# 建立三元词频表
import json

triplet = "{first}{second}{third}"

def extract_triplets(fin, tri_table):
    with open(fin, "r", encoding="utf-8") as file_in:
        while line := file_in.readline():
            for i in range(2, length:=len(line)):
                if (key := triplet.format(first=line[i-2], second=line[i-1], third=line[i])) in tri_table.keys():
                    tri_table[key] += 1
                else:
                    tri_table[key] = 1

if __name__ == "__main__":
    tri_table = {}
    fin = "filtered_{}.txt"
    tables_dir = "../../tables/"
    for j in range(1, 10):
        extract_triplets(fin.format(str(j)), tri_table=tri_table)
        print(j, "done")
    with open(tables_dir+"tri-sina.txt", "w", encoding="utf-8") as file_out:
        file_out.write(json.dumps(tri_table, ensure_ascii=False))