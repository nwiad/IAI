# 建立频率表
import json

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

if __name__ == "__main__":
    sing_table = {}
    dual_table = {}
    fin = "filtered_{}.txt"
    tables_dir = "../../tables/"
    for j in range(1, 10):
        extract(fin.format(str(j)), sing_table=sing_table, dual_table=dual_table)
        print(j, "done")
    with open(tables_dir+"sing-sina.txt", "w", encoding="utf-8") as file_out:
        file_out.write(json.dumps(sing_table, ensure_ascii=False))
        file_out.close()
    with open(tables_dir+"dual-sina.txt", "w", encoding="utf-8") as file_out:
        file_out.write(json.dumps(dual_table, ensure_ascii=False))
        file_out.close()