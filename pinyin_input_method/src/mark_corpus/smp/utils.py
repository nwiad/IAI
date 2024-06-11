import json

char2pinyin = {}
with open("../../tables/char2pinyin.txt", "r", encoding="utf-8") as f:
    char2pinyin = json.loads(f.read())

table = char2pinyin.keys()  # 哈希方法查找

def filter_by_table(fin, fout):
    with open(fout,"w", encoding="utf-8") as file_out:
        with open(fin, "r", encoding="utf-8") as file_in:
            while line := file_in.readline():
                filtered_line = "".join(char for char in line if char in table)
                file_out.write(filtered_line+"\n")

pair = "{first}{second}"

def extract(fin, sing_table, dual_table):
    with open(fin, "r", encoding="utf-8") as file_in:
        while line := file_in.readline():
            if line[0] in sing_table.keys():
                sing_table[line[0]] += 1
            else:
                sing_table[line[0]] = 1

            for i in range(1, length := len(line)):
                if line[i] in sing_table.keys():
                    sing_table[line[i]] += 1
                else:
                    sing_table[line[i]] = 1

                if (key := pair.format(first=line[i-1], second=line[i])) in dual_table.keys():
                    dual_table[key] += 1
                else:
                    dual_table[key] = 1

triplet = "{first}{second}{third}"

def extract_triplets(fin, tri_table):
    with open(fin, "r", encoding="utf-8") as file_in:
        while line := file_in.readline():
            for i in range(2, length:=len(line)):
                if (key := triplet.format(first=line[i-2], second=line[i-1], third=line[i])) in tri_table.keys():
                    tri_table[key] += 1
                else:
                    tri_table[key] = 1