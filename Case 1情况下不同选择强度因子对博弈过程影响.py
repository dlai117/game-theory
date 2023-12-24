# 写代码日期：2023/12/22
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


omega_values = [0.2, 0.4, 0.6, 0.8, 1.0]  # 不同的选择强度因子
epsilon = 0.25
initial_q = 0.1
p11 = p21 = 0.1

# 得益矩阵
d111, d112, d121, d122, d211, d212, d221, d222 = [30, 0, 10, 0, 40, 0, 20, 0]
a111, a112, a121, a122, a211, a212, a221, a222 = [50, -40, 30, -20, 20, -10, 40, -30]


# 防守策略演化方程
def evolution(q, t, w,epsilon, p11, p21):
    dq_dt = w * q * (1 - q) * (
                epsilon * p11 * (d111 + d122 - d112 - d121) + (1 - epsilon) * p21 * (d211 + d222 - d212 - d221) + epsilon * (
                    d121 + d222 - d122 - d221) + d221 - d222)
    return dq_dt


# 时间
t = np.linspace(0, 10, 100)

plt.figure(figsize=(10, 5))

for w in omega_values:
    # 求解不同噪声下的微分方程的解
    q_solution = odeint(evolution, initial_q, t, args=(w, epsilon, p11, p21))
    plt.scatter(t, q_solution[:, 0], marker='.', label=f'ω = {w}')
    plt.plot(t, q_solution[:, 0])

plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('t')
plt.ylabel('q')
plt.title('Evolution of Defense Strategy "Patch Upgrade" with Varying Selection Intensity ω')
plt.legend()
plt.grid(True)
plt.show()
