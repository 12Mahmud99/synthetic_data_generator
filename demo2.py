import matplotlib.pyplot as plt
from generators import *
from utils import *

#print("CUDA device name:", torch.cuda.is_available())
#print(torch.version.cuda)


boltzman= 1.38e-11 
zeta = 2.26e-9 
T = 300
D = (T*boltzman)/zeta
k = 5e-25
t0=0
total_time = 30
time_step = 4e-3 
x_mu = 0
k=2.26e-9 

barrier_height= 1e-8
left_well=-3
right_well=10
tilt=1e-10

pos=torch.arange(-10,10,1e-3)
y=graph_potential(pos,barrier_height,left_well=left_well,right_well=right_well,tilt=tilt)
plt.figure(figsize=(12,4))
plt.plot(pos,y)
#plt.ylim(0,)
plt.xlim(2*left_well,2*right_well)
plt.show()

plt.figure(figsize=(12, 5))

##shallow well particle
p_x0=10
p1,pt=double_wells_generator(total_time=total_time, zeta=zeta,T=T,time_step=time_step,boltzmann=boltzman,right_well=right_well,left_well=left_well,
                          barrier_height=barrier_height,x0_mean=p_x0,device='cpu',num_of_simulations=10,tilt=tilt)

for i in range(p1.shape[0]):
    plt.plot(pt.cpu().numpy(), p1[i].cpu().numpy())
plt.title("Double-well Langevin trajectories")
plt.xlabel("Time")
plt.ylabel("Position")

plt.tight_layout()
plt.show()


##deeper well particle
p_x0=-3
p1,pt=double_wells_generator(total_time=total_time, zeta=zeta,T=T,time_step=time_step,boltzmann=boltzman,right_well=right_well,left_well=left_well,
                          barrier_height=barrier_height,x0_mean=p_x0,device='cpu',num_of_simulations=10,tilt=tilt)

for i in range(p1.shape[0]):
    plt.plot(pt.cpu().numpy(), p1[i].cpu().numpy())
plt.title("Double-well Langevin trajectories")
plt.xlabel("Time")
plt.ylabel("Position")

plt.tight_layout()
plt.show()