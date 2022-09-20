import numpy as np


def fft_ifft(g, x):
    return g * np.fft.fft(np.real(np.fft.ifft(x)) ** 2)


def main():
    # x is used when using a periodic boundary condition, to set up in terms of pi

    # Initial condition and grid setup
    n = 1024  # number of points calculated along x
    x = np.transpose(np.conj(np.arange(1, n + 1))) / n
    u = np.cos(x / 16) * (1 + np.sin(x / 16))  # initial condition
    v = np.fft.fft(u)
    h = 0.25  # time step
    k = np.transpose(np.conj(np.concatenate((np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0))))) / 16
    cap_l = k ** 2 - k ** 4
    e = np.exp(h * cap_l)
    e_2 = np.exp(h * cap_l / 2)
    m = 16
    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    lr = h * np.transpose(np.repeat([cap_l], m, axis=0)) + np.repeat([r], n, axis=0)
    q = h * np.real(np.mean((np.exp(lr / 2) - 1) / lr, axis=1))
    f1 = h * np.real(np.mean((-4 - lr + np.exp(lr) * (4 - 3 * lr + lr ** 2)) / lr ** 3, axis=1))
    f2 = h * np.real(np.mean((2 + lr + np.exp(lr) * (-2 + lr)) / lr ** 3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * lr - lr ** 2 + np.exp(lr) * (4 - lr)) / lr ** 3, axis=1))
    # main loop
    uu = np.array([u])
    tt = 0
    t_max = 150
    n_max = round(t_max / h)
    n_plt = int((t_max / 100) / h)
    g = -0.5j * k
    for n in range(1, n_max + 1):
        t = n * h
        nv = fft_ifft(g, v)
        a = e_2 * v + q * nv
        na = fft_ifft(g, a)
        b = e_2 * v + q * na
        nb = fft_ifft(g, b)
        c = e_2 * a + q * (2 * nb - nv)
        nc = fft_ifft(g, c)
        v = e * v + nv * f1 + 2 * (na + nb) * f2 + nc * f3
        if n % n_plt == 0:
            u = np.real(np.fft.ifft(v))
            uu = np.append(uu, np.array([u]), axis=0)
            tt = np.hstack((tt, t))

    print(uu.shape)
    print(uu)

    np.savetxt('ks.csv', uu.T, delimiter=',', fmt='%.10f')


if __name__ == '__main__':
    main()
