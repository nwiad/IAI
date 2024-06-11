import json

tables_dir = "../tables/"  # 表格目录
triplet = "{first}{second}{third}"  # 三元词
pair = "{first}{second}"  # 二元词

def read_dict(src: str):  # 读取表格
    try:
        with open(src, "r", encoding="utf-8") as f:
            dict = json.loads(f.read())
    except FileNotFoundError as e:
        print(e)
    return dict

pinyin2char = read_dict(tables_dir+"pinyin2char.txt")  # 拼音-汉字表

def check_for_accuracy(output="output.txt", std_output="std_output.txt"):  # 计算准确率
    matched_chars = 0
    matched_lines = 0
    total_chars = 0
    total_lines = 0
    with open(output, "r", encoding="utf-8") as res:
        with open(std_output, "r", encoding="utf-8") as std_res:
            while (line:=res.readline()) and (std_line:=std_res.readline()):
                assert len(line) == len(std_line)
                total_lines += 1
                mismatch = False
                for i in range(0, length:=len(line)):
                    total_chars += 1
                    if line[i] == std_line[i]:
                        matched_chars += 1
                    else:
                        mismatch = True
                if not mismatch:
                    matched_lines += 1
    print("字准确度：{:.2f}%".format(100 * matched_chars/total_chars))
    print("句准确度：{:.2f}%".format(100 * matched_lines/total_lines))