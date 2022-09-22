import numpy as np
from matplotlib.pyplot import *
from math import cos, sin, sqrt, pi, copysign

A = 1
O_m = 3
Th_0 = 0.2

V_MAX = 1.5
AN_MAX = 6
AT_MAX = 10


def y(x: float) -> float:
    return A * sin(O_m * x + Th_0)


def p(x: float) -> float:
    return A * O_m * cos(O_m * x + Th_0)


def q(x: float) -> float:
    return -A * (O_m ** 2) * sin(O_m * x + Th_0)


def k(x: float) -> float:
    return abs(q(x)) / (1 + p(x) ** 2)**(3 / 2)


def vn_max(x: float) -> float:
    return sqrt(AN_MAX / k(x))


def vn_x_max(x: float, dx: float = 1e-6) -> float:
    return (vn_max(x + dx) - vn_max(x)) / dx


# Returns the greatest element that less than given
def lower(elem: float, ls: list[float]) -> int:
    for idx, i in enumerate(ls):
        if i >= elem:
            return idx - 1
    return len(ls) - 1


# Obtain Delta x
def get_crit_dis(dt: float):
    v_crit = 1.441275  # V_entry
    fx_s = 0.34090196  # X_left
    tau = (V_MAX - v_crit) / AT_MAX

    t = 0
    x_c = fx_s
    v_c = v_crit
    # Numeric integration
    while t < tau:
        x_c -= v_c * dt / sqrt(1 + p(x_c) ** 2)
        v_c += AT_MAX * dt
        t += dt

    return fx_s - x_c


# Get 4 - x_stop
def get_end_dis(x_e: float, dt: float):
    tau = V_MAX / AT_MAX

    x_c = x_e
    v_c = 0
    t = 0
    # Numeric integration
    while t < tau:
        x_c -= v_c * dt / sqrt(1 + p(x_c) ** 2)
        v_c += AT_MAX * dt
        t += dt

    return x_e - x_c


# Get velocity dependency from t
def get_vs(x_s: float, x_e: float, dt: float):
    fx_s = 0.34090196  # D' first left border
    fx_e = 0.57296155  # D' first right border
    crit_x = [(fx_s if i % 2 == 0 else fx_e) + (i // 2) * pi / O_m for i in range(4 * 2)]  # D'
    crit_d = get_crit_dis(dt)  # Delta x
    end_d = get_end_dis(x_e, dt)  # 4 - x_stop

    x_c = x_s
    v_c = 0
    cur_a = AT_MAX
    v = []
    t = 0

    # Numerical integration
    while x_c < x_e:
        idx_1 = lower(x_c, crit_x)
        idx_2 = lower(x_c + crit_d, crit_x)

        v.append(v_c)
        if idx_1 % 2 == 0:
            cur_a = AT_MAX
        if idx_2 % 2 == 0 or x_c + end_d >= x_e:
            cur_a = -AT_MAX

        x_c += v_c * dt / sqrt(1 + p(x_c) ** 2)
        if idx_1 % 2 == 0:
            v_c = vn_max(x_c)  # Inside D'
        else:
            if 0 <= v_c + cur_a * dt <= V_MAX:
                v_c += cur_a * dt  # Outside D'
        t += dt

    return v, t


# Get coordinate dependency from t
def get_xyt(v: list[float], dt: float) -> tuple[list[float], list[float]]:
    xst = []
    yst = []
    x_c = 0
    y_c = 0

    for vel in v:
        xst.append(x_c)
        yst.append(y_c)
        x_c += vel * dt / sqrt(1 + p(xst[-1]) ** 2)
        y_c += p(xst[-1]) * vel * dt / sqrt(1 + p(xst[-1]) ** 2)

    return xst, yst


# Filter noice after numerical differentiating
def filter_h_v(ls: list[float], max_val: float, error: float = 1e-4):
    for i, v in enumerate(ls):
        if abs(v - max_val) > error and abs(v) > max_val:
            ls[i] = copysign(max_val, v)


# Obtain acceleration dependency from t
def at_an_t(v: list[float], xs: list[float], dt: float) -> tuple[list[float], list[float]]:
    at = []
    an = []
    dl = 3

    for vel, vel_next in zip(v, v[dl:]):
        at.append((vel_next - vel) / (dt * dl))
    for _ in range(dl):
        at.append(at[-1])

    for vel, x_ in zip(v, xs):
        an.append(k(x_) * vel ** 2)

    filter_h_v(at, AT_MAX)
    filter_h_v(an, AN_MAX)

    return at, an


def main():
    dt = 1e-6
    x_s = 0
    x_e = 4

    v, tau = get_vs(x_s, x_e, dt)  # Tau is total time

    t = np.linspace(0, tau, int(tau / dt))
    xs = np.linspace(x_s, x_e, 1000)
    ys = np.array([y(x) for x in xs])

    # Plots
    title('$y(x)$')
    plot(xs, ys, 'b', label='$y(x)$', linewidth=1.0)
    grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    grid(True)
    xlim([x_s, x_e])
    xticks(np.arange(x_s, x_e, 0.5))
    yticks(np.arange(-1.5, 1.5, 0.2))
    ylabel(r'$y(x), m$')
    xlabel(r'$x, m$')
    legend(loc='best')
    savefig('y_from_x.png')
    show()

    title('$V(t)$')
    plot(t, v, 'b', label='$V(t)$', linewidth=1.0)
    grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    grid(True)
    xlim([0, tau])
    xticks(np.arange(0, tau, 0.5))
    yticks(np.arange(0, 1.5, 0.1))
    ylabel(r'$V(t),\frac{m}{s}$')
    xlabel(r'$t,s$')
    legend(loc='best')
    savefig('vel_from_t.png')
    show()

    xs, ys = get_xyt(v, dt)
    title('$y(t)$, $x(t)$')
    plot(t, xs, 'r', label='$x(t)$', linewidth=1.0)
    plot(t, ys, 'b', label='$y(t)$', linewidth=1.0)
    grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    grid(True)
    xlim([0, tau])
    xticks(np.arange(0, tau, 0.5))
    yticks(np.arange(-1.5, xs[-1], 0.5))
    ylabel(r'$coordinate, m$')
    xlabel(r'$t, s$')
    legend(loc='best')
    savefig('xy_from_t.png')
    show()

    at, an = at_an_t(v, xs, dt)
    title('$a_t(t)$, $a_n(t)$')
    plot(t, at, 'r', label='$a_t(t)$', linewidth=1.0)
    plot(t, an, 'b', label='$a_n(t)$', linewidth=1.0)
    grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    grid(True)
    xlim([0, tau])
    xticks(np.arange(0, tau, 0.5))
    yticks(np.arange(min(at + an), max(at + an), 1))
    ylabel(r'$a, \frac{m}{s^2}$')
    xlabel(r'$t, s$')
    legend(loc='best')
    savefig('acc_from_t.png')
    show()

    print(f'Total time: {tau} s.')


if __name__ == '__main__':
    main()
