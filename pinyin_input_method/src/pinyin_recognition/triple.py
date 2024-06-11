from math import log
import sys
print("Loading...")
from utils import pinyin2char, tables_dir, pair, triplet, read_dict, check_for_accuracy

# 命令行功能
suffix = "-all.txt"
option = True
option_index = -1
argc = len(sys.argv)
if argc < 3 or argc >4:
    print("Bad number of parameters")
    exit()
for i in range(1, argc):
    arg = sys.argv[i]
    if "-" in arg and option:
        option = False
        option_index = i
        length = len(arg)
        if arg[0:3] == "-si" and arg[3:length] == "-sina"[3:length]:
            suffix = "-sina.txt"
        elif arg[0:3] == "-sm" and arg[3:length] == "-smp"[3:length]:
            suffix = "-smp.txt"
        elif arg == "-all"[0:length]:
            suffix = "-all.txt"
        else:
            print("Bad option: "+arg[1:])
            exit()
    elif "-" in arg:
        print("Too many options")
        exit()
input = ""
output = ""
for i in range(1, argc):
    if i != option_index and input == "":
        input = sys.argv[i]
    elif i != option_index:
        output = sys.argv[i]

sing_table = read_dict(tables_dir+"sing"+suffix)  # 字频表

sing_count = 0  # 字的总频数
for key in sing_table.keys():
    sing_count += sing_table[key]

dual_table = read_dict(tables_dir+"dual"+suffix)  # 二元词频表

tri_table = read_dict(tables_dir+"tri"+suffix)  # 三元词频表

dual_comp = 10  # 不存在三元词但存在二元词时的补偿值
sing_comp = 100 # 不存在二元词但存在单字时的补偿值
penalty = 1000  # 结点间不存在路径时的默认耗散值

def expand_first(j):  # 第0层
    sing = pinyin2char[line[0]][j]
    if sing in sing_table:
        return -log(sing_table[sing] / sing_count) + sing_comp
    else:
        return penalty
    
def trivial_get_cost(j, k):  # 第0层的结点到第1层的结点的耗散
    sing = pinyin2char[line[0]][k]
    dual = pair.format(first=pinyin2char[line[0]][k], second=pinyin2char[line[1]][j])
    if dual in dual_table:
        return cost[0][k] - log(dual_table[dual] / sing_table[sing]) + dual_comp
    elif sing in sing_table:
        return cost[0][k] - log(sing_table[sing] / sing_count) + sing_comp
    else:
        return penalty

def trivial_expand(j):  # 确定第1层的第j个结点的前驱以及从起点到此的最小耗散
    index = None
    minimum = None
    for k in range(0, length:=len(pinyin2char[line[0]])):
        tmp = trivial_get_cost(j, k)
        if k == 0:
            index = k
            minimum = tmp
        elif tmp < minimum:
            index = k
            minimum = tmp
    return index, minimum

def get_cost(i, j, k):  # 从起点到第i-1层的第k个结点的最小耗散+第i-1层的第k个结点到第i层的第j个结点的耗散
    first = pinyin2char[line[i-2]][pred[i-1][k]]
    second = pinyin2char[line[i-1]][k]
    third = pinyin2char[line[i]][j]
    tri = triplet.format(first=first, second=second, third=third)
    dual_12 = pair.format(first=first, second=second)
    dual_23 = pair.format(first=second, second=third)
    if tri in tri_table:
        return (cost[i-1][k] - log(tri_table[tri] / dual_table[dual_12]))
    elif dual_23 in dual_table:
        return cost[i-1][k] - log(dual_table[dual_23] / sing_table[second]) + dual_comp
    elif third in sing_table:
        return cost[i-1][k] - log(sing_table[third] / sing_count) + sing_comp
    else:
        return penalty

def expand_node(i, j):  # 确定第i层的第j个结点的前驱和从起点到此的最小耗散
    index = None
    minimum = None
    for k in range(0, length:=len(pinyin2char[line[i-1]])):
        tmp = get_cost(i, j, k)
        if k == 0:
            index = k
            minimum = tmp
        elif tmp < minimum:
            index = k
            minimum = tmp
    return index, minimum

def expand_last(i):  # 确定终点（哨兵结点）的前驱和从起点到此的最小耗散
    index = None
    minimum = None
    for k in range(0, length:=len(pinyin2char[line[i-1]])):
        tmp = cost[i-1][k]
        if k == 0:
            index = k
            minimum = tmp
        elif tmp < minimum:
            index = k
            minimum = tmp
    return index, minimum

def markov():
    for i in range(0, length:=(len(line)+1)):  # 末尾增加一个哨兵
        pred.append([])
        cost.append([])
        if i < length - 1:
            size = len(pinyin2char[line[i]])
        else:
            size = 1
        if i == 0:
            for j in range(0, size):
                minimum = expand_first(j)
                pred[i].append(-1)
                cost[i].append(minimum)
        elif i == 1:
            for j in range(0, size):
                index, minimum = trivial_expand(j)
                pred[i].append(index)
                cost[i].append(minimum)
        elif i < length - 1:
            for j in range(0, size):
                index, minimum = expand_node(i, j)
                pred[i].append(index)
                cost[i].append(minimum)
        else:  # 终点
            index, minimum = expand_last(i)
            pred[i].append(index)
            cost[i].append(minimum)

def recurse():
    recursion = ""
    length = len(line) + 1
    p = pred[length-1][0]
    for i in range(length-1, 0, -1):
        recursion+= pinyin2char[line[i-1]][p]
        p = pred[i-1][p]
    return recursion[::-1]

try:
    with open(output, "w", encoding='utf-8') as fout:
        with open(input, "r") as fin:
            print("Recognizing...")
            first = True
            while line:=fin.readline().split():
                pred = []
                cost = []
                # Markov过程
                markov()
                # 回溯
                res = recurse()
                if first:
                    fout.write(res)
                    first = False
                else:
                    fout.write("\n"+res)
    print("Done")
    # check_for_accuracy()
except FileNotFoundError as e:
    print(e)
except:
    print("Unknown error")