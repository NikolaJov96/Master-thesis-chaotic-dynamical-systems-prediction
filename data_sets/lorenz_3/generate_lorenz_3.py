import numpy as np


def main():

    # Integration time step
    dt = 0.005

    # Number of time steps
    time_steps = int(1e6)

    # Lorenz ODE parameters
    sigma = 10.0
    r = 28.0
    b = 8.0 / 3.0

    # Initial condition
    state = np.array([-10.0, -7.0, 35.0])

    # Store the trajectory
    data = np.zeros((3, time_steps))

    # Integrate the Lorenz ODEs

    for t in range(time_steps):

        if (t + 1) * 100 % time_steps == 0:
            print('\r{}%'.format(int((t + 1) / time_steps * 100)), end='')

        dx_dt = sigma * (state[1] - state[0])
        dy_dt = r * state[0] - state[1] - state[0] * state[2]
        dz_dt = state[0] * state[1] - b * state[2]

        state[0] = state[0] + dx_dt * dt
        state[1] = state[1] + dy_dt * dt
        state[2] = state[2] + dz_dt * dt

        data[:, t] = state[:]

    print()

    np.savetxt('lorenz_3.csv', data.T, delimiter=',', fmt='%.10f')


if __name__ == '__main__':
    main()
