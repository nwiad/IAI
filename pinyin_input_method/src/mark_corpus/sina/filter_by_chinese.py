# 过滤语料库，只保留汉字
import json
import re
month = ["02", "04", "05", "06", "07", "08", "09", "10", "11"]
fin = "2016-{}.txt"
fout = "{}.txt"
for i in range(0,9):
    with open(fout.format(str(i+1)), "w", encoding="utf-8") as file_out:
        with open(fin.format(month[i]), "r", encoding="gbk") as file_in:
            while line := file_in.readline():
                news = json.loads(line)
                title = news["title"]
                content = news["html"]
                filtered_title = re.sub("[^\u4e00-\u9fa5]+", "\n", title)
                filtered_content = re.sub("[^\u4e00-\u9fa5]+", "\n", content)
                file_out.write(filtered_title)
                file_out.write(filtered_content+"\n")
        file_in.close()
    file_out.close()
    print(i+1, "done")