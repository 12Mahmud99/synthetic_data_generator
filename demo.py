import matplotlib.pyplot as plt
from generators import *


boltzman= 1.38e-11#1.38e-23 
zeta = 2.26e-9#2.26e-15  #6 * np.pi * 0.8e-3 * 150e-9 
T = 300#800#298#kelvin
D = (T*boltzman)/zeta#diffusion 
k = 5e-25#5e-15 #n/m^4=kg/(m^3 s^2)=kg/(1e18 micro m^3 s^2)
#x0 = 3 #5e-9
t0=0
total_time = 30
time_step = 4e-3#4e-6 # microsecond 
x_mu = 0
barrier_height= 2e-10#2.5e-8
left_well=10
right_well=-2
x0_mean=left_well

sw,swt=single_well_generator(total_time=total_time, zeta=zeta,T=T,time_step=time_step,x_mu=x_mu,boltzmann=boltzman,k=k,x0_mean=x0_mean,device='cpu')
dw,dwt=double_wells_generator(total_time=total_time, zeta=zeta,T=T,time_step=time_step,boltzmann=boltzman,right_well=right_well,left_well=left_well,
                          barrier_height=barrier_height,x0_mean=x0_mean,device='cpu',num_of_simulations=10)

plt.figure(figsize=(12, 5))
'''
plt.subplot(1, 2, 1)
for i in range(sw.shape[0]):
    plt.plot(swt.cpu().numpy(), sw[i].cpu().numpy())
plt.title("Single-well Langevin trajectories")
plt.xlabel("Time")
plt.ylabel("Position")
'''
plt.subplot(1, 2, 2)
for i in range(dw.shape[0]):
    plt.plot(dwt.cpu().numpy(), dw[i].cpu().numpy())
plt.title("Double-well Langevin trajectories")
plt.xlabel("Time")
plt.ylabel("Position")

plt.tight_layout()
plt.show()