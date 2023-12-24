import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

epsilon = 1 / 4
omega = 1

# 得益矩阵
a111, a112, a121, a122 = 50, -40, 30, -20
d111, d112, d121, d122 = 30, 0, 10, 0


# 复制动态方程
def replicator_dynamics(y, t, epsilon, omega, a11, d11, a12, d21, a22, d22):
    p11, p21, q = y
    dp11dt = omega * p11 * (1 - p11) * (q * (a11 + a22 - a121 - a12) + a12 - a22)
    dp21dt = omega * p21 * (1 - p21) * (q * (a121 + a22 - a121 - a12) + a12 - a22)
    dqdt = omega * q * (1 - q) * ((epsilon * p11 * (d11 + d22 - d112 - d21) +
                                   (1 - epsilon) * p21 * (d21 + d22 - d112 - d21) +
                                   epsilon * (d21 + d22 - d112 - d21) + d21 - d22))
    return [dp11dt, dp21dt, dqdt]


# 初始条件
p11_0 = 0.1
p21_0 = 0.1
q_0 = 0.9
y0 = [p11_0, p21_0, q_0]

# 时间
t = np.linspace(0, 10, 200)

# 求解微分方程
sol = odeint(replicator_dynamics, y0, t, args=(epsilon, omega, a111, d111, a112, d121, a122, d122))

# 结果可视化
plt.plot(t, sol[:, 0], label=f'p11={p11_0} - Risk Type DoS')
plt.plot(t, sol[:, 1], label=f'p21={p21_0} - Conservative Type DoS')
plt.plot(t, sol[:, 2], label=f'q={q_0} - Defender Patch Upgrade')

plt.xlabel('t')
plt.ylabel('Probability')
plt.legend()
plt.title('Evolutionary Dynamics in the Attack-Defense Game')
plt.show()
