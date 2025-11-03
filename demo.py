import matplotlib.pyplot as plt
from generators import *


boltzman= 1.38e-11 
zeta = 2.26e-9 
T = 300
D = (T*boltzman)/zeta
k = 5e-25
t0=0
total_time = 30
time_step = 4e-3 
x_mu = 0
x0_mean=5
k=2.26e-9 

sw,swt=single_well_generator(num_of_simulations=10,total_time=total_time, zeta=zeta,T=T,time_step=time_step,x_mu=x_mu,boltzmann=boltzman,k=k,x0_mean=x0_mean,device='cpu')

plt.figure(figsize=(12, 5))

plt.plot(1, 2, 2)
for i in range(sw.shape[0]):
    plt.plot(swt.cpu().numpy(), sw[i].cpu().numpy())
plt.title("Double-well Langevin trajectories")
plt.xlabel("Time")
plt.ylabel("Position")

plt.tight_layout()
plt.show()


barrier_height= 2e-10
left_well=10
right_well=-2
x0_mean=left_well
dw,dwt=double_wells_generator(total_time=total_time, zeta=zeta,T=T,time_step=time_step,boltzmann=boltzman,right_well=right_well,left_well=left_well,
                          barrier_height=barrier_height,x0_mean=x0_mean,device='cpu',num_of_simulations=10)
