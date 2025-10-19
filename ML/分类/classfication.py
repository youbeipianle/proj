import numpy as np
import matplotlib.pyplot as plt

# 创建变量train，导入.csv文件的数据作为训练数据集
train = np.loadtxt('dataset1.csv', delimiter = ',', skiprows = 1)
train_x = train[:,0:2]
train_y = train[:,2]

# 初始化权重向量，由于输出只有x1和x2两个维度，所以w向量只需要2个参数
w = np.random.rand(2)

# 定义 - 判别函数 f(x)，dot(w,x)大于等于0时函数值取1，反之小于0时，函数值取-1
def f(x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1

# 定义 - 参数更新表达式，f(x)不等于y时，更新 w = w + y * x
#        这里y=1,所以跟新的本质是 w = w + x，向量加法，使w向量旋转
#        或  y=-1, w = w-x, 向量减法，使w向量选择
#                      f(x)等于y时，则不更新 w, w = w

epoch = 5  # 定义 - 循环次数 epoch
count = 0   # 定义 - 更新次数，初始化为0

for _ in range(epoch):  # 用'_'表示这个变量不参与运算，不重要； 也可以用 i 做循环变量
    for x, y in zip(train_x, train_y):
        if f(x) != y:
            w = w + y * x
            count += 1
            print('第{}次, 标签值未匹配，更新！权重 w1 = {:.3f}, w2 = {:.3f}'.format(count, w[0], w[1]))
        else:
            w = w
            count += 1
            print('第{}次，标签值匹配, 不更新！权重 w1 = {:.3f}, w2 = {:.3f}'.format(count, w[0], w[1]))

# 画出分类线，分类线是与向量w垂直的直线，
# 即dot(w,x)=0的直线，w1*x1+w2*x2=0， 即 x2 = -(w1/w2) * x1 这个条线
x1 = np.arange(0, 500) # 生成0至500的等差数组，默认步长为1
plt.plot(x1, -(w[0]/w[1]) * x1, linestyle = 'dashed')

# 画图,取 train_y列中标签值分别为1和-1的行，0取第1列，1取第2列
plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')

#plt.tight_layout()  # 自动调整图片位置
plt.axis('scaled')  # 使坐标轴成比例，避免图失真
plt.show()

# 存图
plt.savefig("fig7.png")
plt.close()




