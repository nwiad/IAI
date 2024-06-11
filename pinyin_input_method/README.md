# README

## 1. 文件结构

```powershell
─data
│      input.txt
│      output.txt（模型：二元模型，过滤策略：将所有非中文字符替换为换行符，语料库：sina + smp）
│
└─src
    ├─mark_corpus
    │  ├─sina
    │  │      build_table.py
    │  │      build_tri_table.py
    │  │      filter_by_chinese.py
    │  │      filter_by_table.py
    │  │
    │  └─smp
    │          add_smp.py
    │          build_smp.py
    │          usual_train_new.txt
    │          utils.py
    │
    ├─mark_pinyin
    │      mark_pinyin.py
    │      拼音汉字表.txt
    │
    ├─pinyin_recognition
    │      dual.py
    │      triple.py
    │      utils.py
    │
    ├─tables（提交时为空目录，需用云盘链接中的同名目录替换）
    │      char2pinyin.txt
    │      dual-all.txt
    │      dual-sina.txt
    │      dual-smp.txt
    │      pinyin2char.txt
    │      sing-all.txt
    │      sing-sina.txt
    │      sing-smp.txt
    │      tri-all.txt
    │      tri-sina.txt
    │      tri-smp.txt
    │
    └─tables(rip all)（提交时为空目录，需用云盘链接中的同名目录替换）
            char2pinyin.txt
            dual-all.txt
            dual-sina.txt
            dual-smp.txt
            pinyin2char.txt
            sing-all.txt
            sing-sina.txt
            sing-smp.txt
            tri-all.txt
            tri-sina.txt
            tri-smp.txt
```

## 2. 拼音汉字表处理程序的运行方式

运行 mark_pinyin 目录下的 mark_pinyin.py ，此程序将读取同一目录下的 拼音汉字表.txt 文件，并在 tables 目录生成 pinyin2char.txt 和 char2pinyin.txt。

## 3. 语料处理程序的运行方式（用时较久、占用空间较大，慎重操作）

语料处理程序位于 mark_corpus 目录，其中 sina 目录中的程序用于处理 sina 语料库，smp 目录中的程序用于处理 smp 语料库。

如果希望单独处理 sina 语料库，首先将语料库中的九个.txt文件复制到 sina 目录下，然后依次运行 filter_by_chinese.py、filter_by_table.py（仅用于压缩词频表大小，耗时较久，可跳过）、build_table.py（生成字频表和二元词频表）、build_tri_table.py（生成三元词频表）。这一流程将读取 sina 语料库的文件并在当前目录生成一系列中间文件，最终在 tables 目录生成 sing-sina.txt、dual-sina.txt、tri-sina.txt。

如果希望单独处理 smp 语料库，smp 语料文件已存在于 smp 目录中，只需运行 build_smp.py ，此程序将读取  usual_train_new.txt 并在 tables 目录生成 sing-smp.txt、dual-smp.txt、tri-smp.txt。

如果希望合并处理 sina 语料库和 smp 语料库，在单独处理 sina 语料库后，运行 smp 目录下的 add_smp.py，该程序将读取 tables 目录下的 sing-sina.txt、dual-sina.txt、tri-sina.txt，并在 tables 目录下生成 sing-all.txt、dual-all.txt、tri-all.txt。

## 4. 拼音识别程序的运行方式

拼音识别程序位于 pinyin_recognition 目录下，dual.py 为二元模型，triple.py 为三元模型。

请事先确保从云盘上获取了 tables 和 tables(rip all) 目录并放置在上述文件结构所示的位置。

### 4.1 过滤策略的选择

拼音识别程序会从名为“tables”的目录加载词频表和拼音汉字表。

有两个过滤策略：

1. 将所有非中文字符替换为换行符
2. 直接去除所有非中文字符

初始情况下，tables 目录存储的是用策略1处理得到的数据，tables(rip all) 目录存储的是用策略2处理得到的数据。

如果选择在处理语料时将所有非中文字符替换为换行符，无需进行任何操作；如果选择在处理语料时直接去除所有非中文字符，则将 tables 目录更名为“tables(sub with enter)”（或者任何不造成冲突的名字），将 tables(rip all) 更名为“tables”（必须为此名称），如需再次切换策略，则恢复二者原本的目录名即可。

### 4.2 命令行运行

在命令行按如下格式输入命令：

```powershell
python <model> <input file> <output file> [corpus]
```

`<model>`为必选参数，应为`dual.py`或`triple.py`。

`<input>`和`<ouput>`为必选参数，分别代表输入和输出文件的路径，且必须输入在前、输出在后。

其中`[corpus]`为可选参数：

- 省略或为`-all`，代表使用 sina + smp 语料库
- 为`-sina`，代表使用 sina 语料库
- 为`-smp`，代表使用 smp 语料库

事实上，`[corpus]`参数可以置于`<moedl>`后的任何位置，并且可以只输入**不致造成混淆的前缀**，如`-a`、`-si`、`-sm`。

若要使用三元模型，将`dual.py`改为`triple.py`即可。

以下是若干合法的命令行示例（假设以 data 目录中的 input.txt 为输入，并将输出文件写入到 data 目录）：

```powershell
python dual.py -a ../../data/input.txt ../../data/output.txt
python dual.pt ../../data/input.txt -sina ../../data/output.txt
puthon triple.py ../../data/input.txt ../../data/output.txt -sm
```

### 4.3 检测准确率

pinyin_recognition/utils.py 中的`check_for_accuracy()`函数可以用于检测字/句准确率，可以通过查看源码了解具体调用方法。