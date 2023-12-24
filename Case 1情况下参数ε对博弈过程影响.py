# 写代码日期：2023/12/22
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


epsilon_values = [0, 0.25, 0.5, 0.75, 1.0]  # 不同的参数ε的取值
Omega = 1  # 选择强度因子
initial_q = 0.1 
p11 = p21 = 0.1 

# 得益矩阵
d111, d112, d121, d122, d211, d212, d221, d222 = [30, 0, 10, 0, 40, 0, 20, 0]
a111, a112, a121, a122, a211, a212, a221, a222 = [50, -40, 30, -20, 20, -10, 40, -30]


# Defense strategies evolution equation
def evolution(q, t, Omega, epsilon, p11, p21):
    dq_dt = Omega * q * (1 - q) * (
               epsilon * p11 * (d111 + d122 - d112 - d121) + (1 - epsilon) * p21 * (d211 + d222 - d212 - d221) + epsilon * (
                    d121 + d222 - d122 - d221) + d221 - d222)
    return dq_dt


# Time points
t = np.linspace(0, 10, 100)

# Plotting
plt.figure(figsize=(10, 5))

for E in E_values:
    # 求解不同噪声下的微分方程的解
    q_solution = odeint(evolution, initial_q, t, args=(Omega, epsilon, p11, p21))
    plt.scatter(t, q_solution[:, 0], marker='.', label=f'epsilon = {epsilon}')
    plt.plot(t, q_solution[:, 0])

plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 1.1, 0.2))
plt.xlabel('t')
plt.ylabel('probability')
plt.title('Evolution of Defense Strategy "Patch Upgrade" with Varying Selection Intensity ω')
plt.legend()
plt.grid(True)
plt.show()
