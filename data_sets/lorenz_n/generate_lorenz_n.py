import numpy as np


def main():

    # Integration time step
    dt = 0.01

    # Number of time steps
    transient_time_steps = int(1000 / dt)
    keeper_time_steps = int(1e6)
    total_time_steps = int(transient_time_steps + keeper_time_steps)

    # Grid (state) size
    grid_size = 40

    # Lorenz ODE parameters
    f = 8

    # Initial condition
    state = np.random.random(grid_size + 3) * 2.0 - 1

    # Store the trajectory
    data = np.zeros((grid_size, keeper_time_steps))

    # Integrate the Lorenz ODEs

    for t in range(total_time_steps):

        if (t + 1) * 100 % total_time_steps == 0:
            print('\r{}%'.format(int((t + 1) / total_time_steps * 100)), end='')

        # Assign helper state variables
        state[0] = state[grid_size]
        state[1] = state[grid_size + 1]
        state[grid_size + 2] = state[2]

        # Calculate steps for actual state variables
        d_dt = (state[3:] - state[:grid_size]) * state[1:grid_size + 1] - state[2:grid_size + 2] + f

        # Update actual state variables
        state[2:grid_size + 2] = state[2:grid_size + 2] + d_dt * dt

        # Save current state actual variables to the array
        if not t < transient_time_steps:
            data[:, t - transient_time_steps] = state[2:grid_size + 2]

    print()

    np.savetxt('lorenz_{}.csv'.format(grid_size), data.T, delimiter=',', fmt='%.10f')


if __name__ == '__main__':
    main()
