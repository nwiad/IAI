# 建立拼音与汉字的双向索引，存储为字典
import json
# 音：字
pinyin2char = {}
# 字：音
char2pinyin = {}
with open("拼音汉字表.txt", "r", encoding="gbk") as f:
    while line := f.readline():
        data = line.split()
        pinyin2char[data[0]] = data[1:]
        for char in data[1:]:
            if char in char2pinyin:
                char2pinyin[char].append(data[0])
            else:
                char2pinyin[char] = [data[0]]
    f.close()
with open("../tables/pinyin2char.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(pinyin2char, ensure_ascii=False))
    f.close()
with open("../tables/char2pinyin.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(char2pinyin, ensure_ascii=False))
    f.close()