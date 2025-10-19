import csv

# 手动输入数据
data = [
    ["x","y"],
    [235,591],
    [216,539],
    [148,413],
    [35,310],
    [85,308],
    [204,519],
    [49,325],
    [25 ,332],
    [173 ,498],
    [191,498],
    [134,392],
    [99 ,334],
    [117,385],
    [112,387],
    [162,425],
    [272,659],
    [159,400],
    [159,427],
    [59,319],
    [198 ,522]
]

# 保存为CSV文件
with open("click.csv","w",newline = "") as file:    # 打开/创建一个click.csv文件
    writer = csv.writer(file)                       # cvs.writer()是python的标准库
    writer.writerows(data)                          # writer.writerows()是CVS标准库    

