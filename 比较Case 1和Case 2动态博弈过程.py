import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# case 2 情况下的得益矩阵（攻击者为risk型）
a111_case2, a112_case2, a121_case2, a122_case2 = 50, -40, 30, -20
d111_case2, d112_case2, d121_case2, d122_case2 = 20, 0, 0, 0
# case 1 情况下的得益矩阵（攻击者为risk型）
a111, a112, a121, a122 = 50, -40, 30, -20
d111, d112, d121, d122 = 30, 0, 10, 0

# 时间
t = np.linspace(0, 10, 200)
epsilon = 1 / 4
omega = 1


# 复制动态方程
def replicator_dynamics(y, t, epsilon, omega, a11, d11, a12, d21, a22, d22):
    p11, p21, q = y
    dp11dt = omega * p11 * (1 - p11) * (q * (a11 + a22 - a121 - a12) + a12 - a22)
    dp21dt = omega * p21 * (1 - p21) * (q * (a121 + a22 - a121 - a12) + a12 - a22)
    dqdt = omega * q * (1 - q) * ((epsilon * p11 * (d11 + d22 - d112 - d21) +
                                   (1 - epsilon) * p21 * (d21 + d22 - d112 - d21) +
                                   epsilon * (d21 + d22 - d112 - d21) + d21 - d22))
    return [dp11dt, dp21dt, dqdt]


# p11 p21 q 的初始值
p11_init = 0.1
p21_init = 0.1
q_init_ = [0.5, 0.9]

# 可视化结果
plt.figure(figsize=(12, 6))
for q_init in q_init_:
    y0 = [p11_init, q_init, q_init]
    sol = odeint(replicator_dynamics, y0, t, args=(epsilon, omega, a111, d111, a112, d121, a122, d122))
    sol_case2 = odeint(replicator_dynamics, y0, t,
                       args=(epsilon, omega, a111_case2, d111_case2, a112_case2, d121_case2, a122_case2, d122_case2))
    plt.plot(t, sol[:, 2], 'b-.', label=f'q={q_init} - Defender Patch Upgrade (Case 1)')
    plt.plot(t, sol_case2[:, 2], 'r-.', label=f'q={q_init} - Defender Patch Upgrade (Case 2)')

plt.xlabel('t')
plt.ylabel('q')
plt.legend(loc='best')
plt.title('Comparative Evolutionary Dynamics (Case 1 vs Case 2)')
plt.grid(True)
plt.show()
