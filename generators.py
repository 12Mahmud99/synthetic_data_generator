import torch
import numpy as np

def single_well_generator(total_time,time_step, x0_mean, zeta, k, x_mu,T,
    boltzmann,num_of_simulations=1,x0_std=0,device='cuda'
):
    '''
    generatres synthetic data for particle(s) following the langevin equation under the force 
    of a single-welled harmonic oscillator

    total_time: total time for each trajectory to run for
    time_step: delta time
    x0_mean: the mean of the initial position's distribution 
    zeta: damping coefficient
    k: force constant
    x_mu: center of the well
    T: termparature
    boltzmann: boltzmann constant (1.38e-23 in SI units)
    num_of_simulations: number of trajectories to produce
    x0_std: the standard deviation of the standard normal gaussian the initial position is sampled from
    device: uss cpu (default device)
    '''

    torch.set_default_device(device)
    if x0_std == 0:
        x0 = torch.full((num_of_simulations,), x0_mean, device=device)
    else:
        x0 = torch.normal(mean=torch.tensor(x0_mean), std=torch.tensor(x0_std), size=(num_of_simulations,))  # sample the initial position

    num_time_steps = int(total_time/time_step)  # calculates the number of time steps given the time

    diffusion = torch.sqrt(torch.tensor(2*boltzmann*T/zeta,device=device))
    dt_tensor = torch.tensor(time_step, device=device)
    positions = torch.zeros((num_of_simulations,num_time_steps),device=device)  # use default float32
    positions[:, 0] = x0
    times = torch.arange(0, total_time, time_step, device=device)

    dw = torch.normal(mean=0, std=np.sqrt(time_step), size=(num_of_simulations, num_time_steps-1),device=device)  # sample brownian motion increments

    for i in range(1, num_time_steps):
        positions[:, i] = positions[:, i-1] + -(k/zeta) * (positions[:, i-1] - x_mu) * dt_tensor + diffusion * dw[:, i-1]

    return (positions, times)

def double_wells_generator(total_time,time_step, x0_mean, zeta, left_well, right_well, barrier_height,T,boltzmann,tilt=0,num_of_simulations=1,x0_std=0,device='cuda'):
        '''
        left_well: position of the left well
        right_well:position of the right well
        '''
        
        midpoint = (left_well + right_well)/2.0 
        a = abs(left_well-right_well)/2.0  

        torch.set_default_device(device)
        if x0_std == 0:
            x0 = torch.full((num_of_simulations,), x0_mean, device=device)
        else:
            x0 = torch.normal(mean=torch.tensor(x0_mean), std=torch.tensor(x0_std), size=(num_of_simulations,))  # sample the initial position

        num_time_steps = int(total_time/time_step)  # calculates the number of time steps given the time

        diffusion = torch.sqrt(torch.tensor(2*boltzmann*T/zeta,device=device))
        dt_tensor = torch.tensor(time_step, device=device)
        positions = torch.zeros((num_of_simulations,num_time_steps),device=device)  # use default float32
        positions[:, 0] = x0
        times = torch.arange(0, total_time, time_step, device=device)
        dw = torch.normal(mean=0, std=np.sqrt(time_step), size=(num_of_simulations, num_time_steps-1),device=device)  # sample brownian motion increments

        for i in range(1, num_time_steps):
            x = positions[:,i-1]
            dUdx = barrier_height*(4*((x - midpoint)**3)/(a**4) - 4*(x - midpoint)/(a**2)) +tilt

            positions[:,i] = x - (dUdx/zeta)*dt_tensor + diffusion*dw[:,i-1]

        return (positions,times)

def single_well_generator_underdamped(total_time, time_step, x0_mean, v0_mean, zeta, m, k, x_mu, T,
                                      boltzmann, num_of_simulations=1, x0_std=0, v0_std=0, device='cuda'):
    """
    generates simulations for a single-well harmonic oscillator
    
    v0_mean: initial velocity mean
    v0_std: initial velocity standard deviation
    m: particle mass
    """
    torch.set_default_device(device)

    if x0_std == 0:
        x = torch.full((num_of_simulations,), x0_mean, device=device)
    else:
        x = torch.normal(mean=torch.tensor(x0_mean), std=torch.tensor(x0_std), size=(num_of_simulations,), device=device)

    if v0_std == 0:
        v = torch.full((num_of_simulations,), v0_mean, device=device)
    else:
        v = torch.normal(mean=torch.tensor(v0_mean), std=torch.tensor(v0_std), size=(num_of_simulations,), device=device)

    num_time_steps = int(total_time / time_step)
    positions = torch.zeros((num_of_simulations, num_time_steps), device=device)
    velocities = torch.zeros((num_of_simulations, num_time_steps), device=device)
    positions[:, 0] = x
    velocities[:, 0] = v
    times = torch.arange(0, total_time, time_step, device=device)

    diffusion = torch.sqrt(torch.tensor(2 * zeta * boltzmann * T / time_step, device=device))
    dt_tensor = torch.tensor(time_step, device=device)

    dw = torch.normal(mean=0, std=np.sqrt(time_step), size=(num_of_simulations, num_time_steps-1), device=device)

    for i in range(1, num_time_steps):
        F = -k * (x - x_mu)
        a = (F - zeta * v) / m
        v = v + a * dt_tensor + diffusion / m * dw[:, i-1]
        x = x + v * dt_tensor

        positions[:, i] = x
        velocities[:, i] = v

    return positions, velocities, times


def double_wells_generator_underdamped(total_time, time_step, x0_mean, v0_mean, zeta, m,
                                       left_well, right_well, barrier_height, T, boltzmann,
                                       tilt=0, num_of_simulations=1, x0_std=0, v0_std=0, device='cuda'):
    """
    underdamped Langevin for a double-well potential
    """
    midpoint = (left_well + right_well) / 2.0
    a = abs(left_well - right_well) / 2.0

    torch.set_default_device(device)

    if x0_std == 0:
        x = torch.full((num_of_simulations,), x0_mean, device=device)
    else:
        x = torch.normal(mean=torch.tensor(x0_mean), std=torch.tensor(x0_std), size=(num_of_simulations,), device=device)

    if v0_std == 0:
        v = torch.full((num_of_simulations,), v0_mean, device=device)
    else:
        v = torch.normal(mean=torch.tensor(v0_mean), std=torch.tensor(v0_std), size=(num_of_simulations,), device=device)

    num_time_steps = int(total_time / time_step)
    positions = torch.zeros((num_of_simulations, num_time_steps), device=device)
    velocities = torch.zeros((num_of_simulations, num_time_steps), device=device)
    positions[:, 0] = x
    velocities[:, 0] = v
    times = torch.arange(0, total_time, time_step, device=device)

    diffusion = torch.sqrt(torch.tensor(2 * zeta * boltzmann * T / time_step, device=device))
    dt_tensor = torch.tensor(time_step, device=device)
    dw = torch.normal(mean=0, std=np.sqrt(time_step), size=(num_of_simulations, num_time_steps-1), device=device)

    for i in range(1, num_time_steps):
        dUdx = barrier_height*(4*((x - midpoint)**3)/(a**4) - 4*(x - midpoint)/(a**2)) + tilt
        a = (-dUdx - zeta * v) / m
        v = v + a * dt_tensor + diffusion / m * dw[:, i-1]
        x = x + v * dt_tensor

        positions[:, i] = x
        velocities[:, i] = v

    return positions, velocities, times
