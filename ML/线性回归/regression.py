import numpy as np
import matplotlib.pyplot as plt
from Plot import train_x
from Plot import train_y

# 参数初始化
theta0 = np.random.rand()
theta1 = np.random.rand()

# 定义-预测函数
def f(x):
    return theta0 + theta1 * x

# 定义-目标函数E(theta)
def E(x,y):
    return 0.5 * np.sum((y - f(x)) ** 2)    # **n 表示 n次方

# 标准化,预处理训练数据
mu = train_x.mean()
sigma = train_x.std()

def standardize(x):         # 定义-标准化的计算公式
    return (x - mu)/sigma   # def函数后要给返回值 return

train_z = standardize(train_x)

# 画图
plt.plot(train_z, train_y, 'o')
#plt.show()

# 保存图形为文件
plt.savefig("fig2.png")  # 保存为 PNG 文件
plt.close()  # 关闭图形窗口

ETA = 1e-3      # 定义-学习率，初值步长设 0.001
diff = 1        # 定义-误差，初值
count = 0       # 定义-运算次数，初值

error = E(train_z, train_y) # 计算所有数据的平方和，E(theta)函数在regeression中定义了

while diff > 1e-2:           # 定义误差阈值
    temp0 = theta0 - ETA * np.sum(f(train_z)-train_y)       # 计算参数1，存于临时变量temp0
    temp1 = theta1 - ETA * np.sum((f(train_z)-train_y) * train_z)   # 计算参数2，存于临时变量temp1

    theta0 = temp0  # 更新参数theta
    theta1 = temp1

    current_error = E(train_z, train_y) # theta更新后，E()函数也同时改变
    diff = error - current_error        # 计算每次误差变化的该变量
    error = current_error               # 更新误差error

    count += 1  # 更新次数加1
    # 定义-log字符串变量，定义日志模板格式，{ }是占位符，里面的:表示格式标识的开始
    log = '第{}次，theta0 = {:3f}，theta1 = {:3f}，误差 = {:4f}，误差步长 = {:4f}'
    print(log.format(count, theta0, theta1, error, diff))

# 画图 - 找到参数theta后，则预测函数 f(x)=theta0+theta1*x 也被确定
plt.plot(train_z, train_y, 'o')
x = np.linspace(-3, 3, 100)      # [-3,3]之间等间距创建100个点，横坐标
plt.plot(x, f(x))
plt.show()

# 保存图形为文件
plt.savefig("fig3.png")  # 保存为 PNG 文件
plt.close()  # 关闭图形窗口
