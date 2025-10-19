import numpy as np
import matplotlib.pyplot as plt

# 导入CSV数据
train = np.loadtxt('/home/zz/proj/ML/线性回归/click.csv', delimiter = ',', skiprows = 1)
train_x = train[:,0]    # 读取第1行 
train_y = train[:,1]    # 读取第2行

mu = np.mean(train_x)   # 期望
sigma = np.std(train_x)  # 标准差（方差开方）

def standardize(x):     # 定义 - 标准化处理函数
    return (x - mu)/sigma

train_z = standardize(train_x)   # 对train_x做标准化处理

# 初始化参数 theta，3个，假设预设函数不再是直线，是曲线 f(x)=theta0 + theta1*x + theta2*x^2
theta = np.random.rand(3)   # 随机生成3个theta

# 定义-一个函数，接收一个x，返回一个矩阵
def to_matrix(x):
    return np.vstack((np.ones(x.shape[0]), x, x**2)).T
# x.shape -输出x的维度，一维数组输入长度，二维则是行和列，三维输出行列和深度
# vstack - 将接收的数组竖着叠起来，这里堆叠后的矩阵是
#          [[1,    1,    1,     ...  1,      1 ]     多少个1由x.shape决定
#           [x1,   x2,   x3,    ...  xn-1,   xn]     数组 x 直接放第2行
#           [x1^2, x2^2, x3^2,  ... ,xn-1^2, xn^2]]
# .T - 转置

X = to_matrix(train_z)  # 训练数据集，矩阵化

# 定义 - 预测函数 f(x)=theta0 + theta1*x + theta2*x^2
def f(x):
    return np.dot(x, theta)     # x是 n行3列，theta是1行3列， 结果是 n行1列

# 定义-目标函数E(theta)
def E(x,y):
    return 0.5 * np.sum((y - f(x)) ** 2)    # **n 表示 n次方

# 定义 - 均方误差 MSE
def MSE(x, y):
    return 1/x.shape[0] * np.sum((f(x) - y)**2)

ETA = 1e-3  # 更新步长
diff = 1    # 误差变化

count = 0

for i in range(3):
    p = np.random.permutation(X.shape[0])     # 随机调换矩阵X的行的顺序，p作为行的索引，

    X = X[p]
    train_y = train_y[p]

    error = E(X, train_y)   #误差
    errors = []

    while diff > 1e-2:
        # 更新 theta
        theta = theta - ETA * np.dot(f(X) - train_y, X)
        
        current_error = E(X, train_y)   # 更新后，再计算一次误差
        diff = error - current_error    # 计算误差消除的步长
        error = current_error           # 更新误差值

        errors.append(MSE(X, train_y))

        count += 1  # 更新次数加1
        # 定义-log字符串变量，定义日志模板格式，{ }是占位符，里面的:表示格式标识的开始
        log = '第{}次，theta0 = {:3f}，theta1 = {:3f}，theta2 = {:3f}，误差 = {:4f}，误差步长 = {:4f}'
        print(log.format(count, theta[0], theta[1], theta[2], error, diff))

    # 画图， MSE的变化
    x_range = np.arange(len(errors))
    plt.plot(x_range, errors, label='curve{}'.format(i))
   

plt.tight_layout()  # 自动调整布局
plt.legend()        # 添加图例
plt.show()

# 保存图形为文件
plt.savefig("fig6.png")  # 保存为 PNG 文件
plt.close()  # 关闭图形窗口