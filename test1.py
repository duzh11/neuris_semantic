import torch
import matplotlib.pyplot as plt

# 定义公式
prob_2_uncertainty = lambda x: -torch.log(x + 1e-12) * x

# 生成 x 值
x_values = torch.linspace(0, 1, 100)  # 从0到1生成100个值

# 计算 y 值
y_values = prob_2_uncertainty(x_values)

# 绘制曲线
plt.plot(x_values.numpy(), y_values.numpy())
plt.xlabel('x')
plt.ylabel('y')
plt.title('Probability to Uncertainty Function')
plt.show()
