# 再次过滤语料库，只保留一二级汉字表中的汉字
import json

char2pinyin = {}
with open("../../old_tables/char2pinyin.txt", "r", encoding="utf-8") as f:
    char2pinyin = json.loads(f.read())

table = char2pinyin.keys()  # 哈希方法查找
fin = "{}.txt"
fout = "filtered_{}.txt"

def filter_by_table(fin, fout):
    with open(fout,"w", encoding="utf-8") as file_out:
        with open(fin, "r", encoding="utf-8") as file_in:
            while line := file_in.readline():
                filtered_line = "".join(char for char in line if char in table)
                file_out.write(filtered_line+"\n")

if __name__ == "__main__":
    for i in range(1, 10):
        filter_by_table(fin.format(str(i)), fout.format(str(i)))
        print(i, "done")