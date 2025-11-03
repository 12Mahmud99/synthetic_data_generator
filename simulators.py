import matplotlib.pyplot as plt
import torch
from abc import ABC, abstractmethod
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from math import sqrt


class LangevinDynamicsSimulator(ABC):
    def __init__(self,m, x0, t0, K_b, T, lambda_, device,delta_t,total_time,v0):
        self.m = m #mass (kg)
        self.x0 = x0 #initial position (m)
        self.t0 = t0 #initial position (s)
        self.K_b = K_b #Boltzman constant (J/K)
        self.T = T #temperature (K)
        self.lambda_ = lambda_ #damping coefficient (Kg/s)
        self.device = device
        self.delta_t = delta_t #time step (s)
        self.all_velocities = None
        self.D = (K_b * T) / lambda_ 
        self.total_time=total_time

        self.steps = 0
        self.num_simulations = 0 
        self.x = x0 #latest position x(t)
        self.v0=v0
        self.all_positions = None  #rows (sim), col (time)
        self.all_times = None #times for each x(t)
        self.t = 0

    def initialize_simulation(self, num_simulations, steps):
        self.num_simulations = num_simulations 
        self.steps = int(self.total_time/self.delta_t)

        self.x = torch.ones(self.num_simulations, device=self.device) * self.x0 
        self.all_positions = torch.zeros((self.num_simulations, self.steps), device=self.device)
        self.all_times = torch.zeros(self.steps, device=self.device)

    @abstractmethod
    def plot(self, simulations_to_plot=None):
        pass

    @abstractmethod
    def run_overdamped_simulations(self, num_simulations, steps):
        pass

    @abstractmethod
    def run_underdamped_simulations(self, num_simulations, steps):
        pass



class SingleDimHarmonicLangevinSimulator(LangevinDynamicsSimulator):
    def __init__(self, m, x0, t0, K_b, T, lambda_, k, total_time, v0, device, delta_t,x_mu):
        super().__init__(m, x0, t0, K_b, T, lambda_, device, delta_t, total_time, v0)
        self.k = k #spring constant (kg/s^2) or (N/m)
        self.x_mu = x_mu#center of harmonic_trap
        
    def find_p_x_t_analytically(self,steps=0,plot=True):
        '''
        calculates p(x,t) via solving the fokker plank equation
        '''
        t = torch.arange(steps, dtype=torch.float32) * self.delta_t 
        variance = ((self.K_b * self.T)/self.k)*(1-torch.exp((-2*self.K*t)/(self.lambda_)))
        mean = self.x0 * torch.exp((-self.k * t)/(self.lambda_))
        x = x.unsqueeze(0) 
        mean = mean.unsqueeze(1)
        variance = variance.unsqueeze(1) 
        coef = 1.0 / torch.sqrt(2 * torch.pi * variance) 
        exponent = - (x - mean) ** 2 / (2 * variance) 
        p_x_t = coef * torch.exp(exponent)
        if plot:
            plt.figure(figsize=(8, 5))
            for i in range(steps):
                plt.plot(x.numpy(), p_x_t[i].numpy(), label=f't={t[i].item():.2f}, μ={mean_t[i]:.2f}, σ={var_t[i].sqrt():.2f}')
            plt.xlabel('x')
            plt.ylabel('p(x,t)')
            plt.title('Fokker-Planck solution in harmonic potential')
            plt.legend()
            plt.grid(True)
            plt.show()

        return p_x_t
    
    ####deprecated
    def run_simulations(self, num_simualtions=0):
        if self.m == 0:
            return self.run_overdamped_simulations(num_simulations=num_simualtions)
        else:
            return self.run_underdamped_simulations(num_simulations=num_simualtions)
        
    def run_overdamped_simulations(self, num_simulations=0,initial_sigma=0.6):
        '''assert num_simulations > 0, "Number of simulations must be > 0"
        assert type(num_simulations) == int, "Number of simulations must be an integer"
        assert steps > 0, "Number of steps must be > 0"
        assert type(steps) == int, "Number of steps must be an integer"
        '''
       
        self.steps = int(self.total_time / self.delta_t)  # Number of time steps
        self.t = self.t0
        
        sigma = torch.sqrt(torch.tensor((2 * self.K_b * self.T)/(self.lambda_) ))
        dt_tensor = torch.tensor(self.delta_t, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)

        dW = sqrt_dt * torch.randn(num_simulations, self.steps - 1, device=self.device)

        self.all_positions = torch.zeros((num_simulations, self.steps), device=self.device)

        #print(self.x_mu)

        self.all_times=torch.zeros(self.steps,device=self.device)
        self.all_times[0] = self.t0

        self.all_positions[:,0] =  self.x0+ initial_sigma* torch.randn_like(self.all_positions[:,0])

        for i in range(1,self.steps): 
            self.all_positions[:,i] = self.all_positions[:,i-1] + \
                  -(self.k/self.lambda_) * (self.all_positions[:,i-1] - self.x_mu) * dt_tensor +\
                  sigma * dW[:,i-1] 

            self.t += self.delta_t
            self.all_times[i] = self.t

        #print(self.all_positions)

        return (torch.max(self.all_positions), 
                torch.min(self.all_positions),
                self.t, 
                self.t0)
    
    def run_overdamped_simulations_double_wells(self, num_simulations=0,initial_sigma=0.6,wells_distance=0):
        '''assert num_simulations > 0, "Number of simulations must be > 0"
        assert type(num_simulations) == int, "Number of simulations must be an integer"
        assert steps > 0, "Number of steps must be > 0"
        assert type(steps) == int, "Number of steps must be an integer"
        '''
       
        self.steps = int(self.total_time / self.delta_t)  # Number of time steps
        self.t = self.t0
        
        sigma = torch.sqrt(torch.tensor((2 * self.K_b * self.T)/(self.lambda_) ))
        dt_tensor = torch.tensor(self.delta_t, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)

        dW = sqrt_dt * torch.randn(num_simulations, self.steps - 1, device=self.device)

        self.all_positions = torch.zeros((num_simulations, self.steps), device=self.device)

        #print(self.x_mu)

        self.all_times=torch.zeros(self.steps,device=self.device)
        self.all_times[0] = self.t0

        self.all_positions[:,0] =  self.x0+ initial_sigma* torch.randn_like(self.all_positions[:,0])

        for i in range(1,self.steps): 
            self.all_positions[:,i] = self.all_positions[:,i-1] + \
                  -(self.k/self.lambda_) * 4*(self.all_positions[:,i-1] - self.x_mu)*((self.all_positions[:,i-1] - self.x_mu)**2 -\
                     wells_distance**2) * dt_tensor +\
                  sigma * dW[:,i-1]

            self.t += self.delta_t
            self.all_times[i] = self.t

        #print(self.all_positions)

        return (torch.max(self.all_positions), 
                torch.min(self.all_positions),
                self.t, 
                self.t0)
    
    def run_overdamped_simulations_double_wells_asymmetric(self, x_L, x_R, h,s, num_simulations=0, initial_sigma=0.6):
        '''
        geometricall: h is the depth of the wells or the height of the energy barrier
        x_L is the position of the left well
        x_R is the position of the right well
        '''
        
        x_c = (x_L + x_R)/2.0 #midpoint
        a = (x_R - x_L)/2.0  #halg distance between wells or from well and midpoint

        b = h*(4*((x_L - x_c)**3)/a**4 - 4*(x_L - x_c)/a**2) #where the potential is minimum at x_L
        print(f'a={a}, b={b}')

        self.steps = int(self.total_time / self.delta_t)
        self.t = self.t0

        sigma = torch.sqrt(torch.tensor((2 * self.K_b * self.T)/(self.lambda_)))
        dt_tensor = torch.tensor(self.delta_t, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)
        dW = sqrt_dt * torch.randn(num_simulations, self.steps - 1, device=self.device)

        self.all_positions = torch.zeros((num_simulations, self.steps), device=self.device)
        self.all_times = torch.zeros(self.steps, device=self.device)
        self.all_times[0] = self.t0
        self.all_positions[:,0] = self.x0 + initial_sigma * torch.randn_like(self.all_positions[:,0])

        for i in range(1, self.steps):
            x = self.all_positions[:,i-1]
            #dUdx = h*(4*((x - x_c)**3)/(a**4) - 4*(x - x_c)/(a**2)) - b
            dUdx = h*(4*((x - x_c)**3)/(a**4) - 4*(x - x_c)/(a**2)) + s

            self.all_positions[:,i] = x - (dUdx/self.lambda_)*dt_tensor + sigma*dW[:,i-1]
            self.t += self.delta_t
            self.all_times[i] = self.t

        return (torch.max(self.all_positions),
                torch.min(self.all_positions),
                self.t,
                self.t0)

    def plot_double_well_asymmetric(self, x_L, x_R, h, num_points=500,a=None,b=None,s=0):
        x_c = (x_L + x_R) / 2.0    # midpoint
        if a is None:
            a = (x_R - x_L) / 2.0      # half distance between wells
        if b is None:
            b = h * (4*((x_L - x_c)**3)/a**4 - 4*(x_L - x_c)/a**2)
        print(f'a={a}, b={b}')

        def U(x,s=s):
            return h * (((x - x_c)**2) / (a**2) - 1)**2 - b * (x - x_c)
            #return h*(((x - x_c)/a)**4-2*((x - x_c)/a)**2)-(b/a)*(x - x_c)

        x_vals = torch.linspace(x_L - 2*a, x_R + 2*a, num_points)
        U_vals = U(x_vals)

        plt.figure(figsize=(6,4))
        plt.plot(x_vals, U_vals, label='Asymmetric double-well', color='blue')
        plt.axvline(x_L, color='red', linestyle='--', label='Left well')
        plt.axvline(x_R, color='green', linestyle='--', label='Right well')
        plt.axvline(x_c, color='gray', linestyle=':', label='Center')
        plt.ylim(0,h*2)
        plt.xlabel('x (μm)')
        plt.ylabel('U(x) (J)')
        plt.title('Asymmetric Quartic Doubble-Well Potential')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_double_well_asymmetric_geoemtry(self, x_L, x_R, h,a,b, num_points=500):
        x = torch.linspace(self.x_mu-1e10,self.x_mu+1e10, num_points)
        U = h*(((x-self.x_mu)/a)**4-2*((x-self.x_mu)/a)**2)-(b/a)*(x-self.x_mu)
    
        plt.figure(figsize=(6,4))
        plt.plot(x, U, label='Asymmetric double-well', color='blue')
        plt.xlabel('x (μm)')
        plt.ylabel('U(x) (J)')
        plt.title('Asymmetric Quartic Doubble-Well Potential')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_overdamped_simulations_double_wells_quartic(self, x_L, x_R, h, num_simulations=0, initial_sigma=0.6):
        """
        Langevin simulation with asymmetric double-well potential in quartic form.

        x_L : position of left well
        x_R : position of right well
        h   : well depth or energy scale
        """

        x_mu = (x_L + x_R)/2.0        # center of wells
        d = (x_R - x_L)/2.0           # half-distance between wells
        k = h / d**4

        b = 4*k*d*(x_mu - x_L) * (d - (x_L - x_mu)**2 / d) 
        self.steps = int(self.total_time / self.delta_t)
        self.t = self.t0

        sigma = torch.sqrt(torch.tensor((2 * self.K_b * self.T)/(self.lambda_)))
        dt_tensor = torch.tensor(self.delta_t, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)
        dW = sqrt_dt * torch.randn(num_simulations, self.steps - 1, device=self.device)

        self.all_positions = torch.zeros((num_simulations, self.steps), device=self.device)
        self.all_times = torch.zeros(self.steps, device=self.device)
        self.all_times[0] = self.t0
        self.all_positions[:,0] = self.x0 + initial_sigma * torch.randn_like(self.all_positions[:,0])

        for i in range(1, self.steps):
            x = self.all_positions[:,i-1]
            dUdx = 4 * k * (x - x_mu) * ((x - x_mu)**2 - d**2) - b

            self.all_positions[:,i] = x - (dUdx/self.lambda_)*dt_tensor + sigma*dW[:,i-1]
            self.t += self.delta_t
            self.all_times[i] = self.t

        return (torch.max(self.all_positions),
                torch.min(self.all_positions),
                self.t,
                self.t0)
    
    def plot_double_well_asymmetric_quartic(self,x_L, x_R, h, num_points=500):
        x_mu = (x_L + x_R)/2.0  
        d = (x_R - x_L)/2.0      
        k = h / d**4             
        b = 4 * k * (x_L - x_mu) * ((x_L - x_mu)**2 - d**2)

        x_vals = torch.linspace(x_L - 2*d, x_R + 2*d, num_points)
        U_vals = k * ((x_vals - x_mu)**2 - d**2)**2 - b * (x_vals - x_mu)

        plt.figure(figsize=(6,4))
        plt.plot(x_vals, U_vals, label='Asymmetric double-well', color='blue')
        plt.axvline(x_L, color='red', linestyle='--', label='Left well')
        plt.axvline(x_R, color='green', linestyle='--', label='Right well')
        plt.axvline(x_mu, color='gray', linestyle=':', label='Center')
        plt.xlabel('x (μm)')
        plt.ylabel('U(x) (J)')
        plt.title('Asymmetric Quartic Double-Well Potential')
        plt.legend()
        plt.grid(True)
        plt.show()



        
    def run_underdamped_simulations(self, num_simulations,initial_sigma):
        '''assert num_simulations > 0, "Number of simulations must be > 0"
        assert type(num_simulations) == int, "Number of simulations must be an integer"
        assert steps > 0, "Number of steps must be > 0"
        assert type(steps) == int, "Number of steps must be an integer"'''

        '''tau = self.m/self.lambda_
        self.total_time *= tau 
        self.delta_t *= tau '''
        
        self.steps = int(self.total_time / self.delta_t)  # Number of time steps\

        #sigma = torch.sqrt(torch.tensor((2 * self.lambda_ * self.K_b * self.T)/(self.m **2) ))
        sigma = torch.sqrt(torch.tensor(2 * self.lambda_ * self.K_b * self.T))/(self.m)
        dt_tensor = torch.tensor(self.delta_t, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)

        self.all_velocities = torch.zeros((num_simulations, self.steps), device=self.device)
        dW = sqrt_dt * torch.randn(num_simulations, self.steps - 1, device=self.device)

        self.all_positions = torch.zeros((num_simulations, self.steps), device=self.device)
        self.all_velocities = torch.zeros((num_simulations, self.steps), device=self.device)

        #print(self.x_mu_1)

        self.all_times=torch.zeros(self.steps,device=self.device)
        self.all_times[0] = self.t0

        self.all_positions[:,0] =  self.x0+ initial_sigma* torch.randn_like(self.all_positions[:,0])
        #self.x0+ .6* torch.randn_like(self.all_positions[:,0])#self.x0 
        # or use 
        self.all_velocities[:,0] = self.v0+ initial_sigma* torch.randn_like(self.all_velocities[:,0])
        #self.v0

        for i in range(1,self.steps): 
            '''self.all_velocities[:, i] = self.all_velocities[:, i-1] \
                + ((-self.lambda_ * self.all_velocities[:, i-1] \
                - self.k * (self.all_positions[:, i-1] - self.x_mu)) * dt_tensor) / self.m \
                + sigma * dW[:, i-1]'''
            
            self.all_velocities[:, i] = self.all_velocities[:, i-1] \
                + (-self.lambda_ * self.all_velocities[:, i-1] \
                - self.k * (self.all_positions[:, i-1] - self.x_mu)) * (dt_tensor/self.m) \
                + sigma * dW[:, i-1]

            #v += ((-self.lambda_ * v - self.k * (self.x - self.x0)) * (self.delta_t))/self.m + sigma* dW
            self.all_positions[:,i] = self.all_positions[:,i-1] + self.all_velocities[:,i-1] * dt_tensor

            self.t += self.delta_t
            self.all_times[i] = self.t

        #print(self.all_positions)

        return (torch.max(self.all_positions), 
                torch.min(self.all_positions),
                self.t, 
                self.t0)
    def plot_tc(self, num_simulations=1, tau=None):
        time = torch.linspace(0, self.total_time, self.steps, device=self.device)
        if tau == None:
            tau = self.m / self.lambda_
        

        X_np = self.all_positions.detach().cpu().numpy()
        plt.figure(figsize=(12, 4))
        plt.title("Position x(t) over Time")
        plt.xlabel("Time (in units of τ)")
        plt.ylabel("Position x(t)")
        for j in range(num_simulations):
            plt.plot(time / tau, X_np[j])
        plt.axhline(self.x_mu,color='yellow')
        plt.grid(True)
        plt.legend()
        plt.show()

        if self.all_velocities != None: 
            V_np = self.all_velocities.detach().cpu().numpy()
            plt.figure(figsize=(12, 4))
            plt.title("Velocity v(t) over Time")
            plt.xlabel("Time (in units of τ)")
            plt.ylabel("Velocity v(t)")
            for j in range(num_simulations):
                plt.plot(time / tau, V_np[j])
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Optional
            plt.grid(True)
            plt.legend()
            plt.show()

    def plot(self, num_simulations=1, tau=None):
        time = torch.linspace(0, self.total_time, self.steps, device=self.device)
        time_np = time.detach().cpu().numpy()

        X_np = self.all_positions.detach().cpu().numpy()
        plt.figure(figsize=(12, 4))
        plt.title("Position x(t) over Time")
        plt.xlabel("Time")
        plt.ylabel("Position x(t)")
        for j in range(num_simulations):
            plt.plot(time_np, X_np[j])
        plt.axhline(self.x_mu,color='yellow')
        plt.grid(True)
        plt.legend()
        plt.show()

        if self.all_velocities != None: 
            V_np = self.all_velocities.detach().cpu().numpy()
            plt.figure(figsize=(12, 4))
            plt.title("Velocity v(t) over Time")
            plt.xlabel("Time")
            plt.ylabel("Velocity v(t)")
            for j in range(num_simulations):
                plt.plot(time, V_np[j])
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Optional
            plt.grid(True)
            plt.legend()
            plt.show()   

    def plot_animation(self, num_simulations=1, interval=30):
        time = torch.linspace(0, self.total_time, self.steps, device=self.device)
        X_np = self.all_positions.detach().cpu().numpy()
        time_np = time.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title("Position x(t) over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position x(t)")
        ax.grid(True)

        # fix axis limits
        ax.set_xlim(time_np.min(), time_np.max())
        ax.set_ylim(X_np.min(), X_np.max())
        ax.axhline(self.x_mu)

        lines = []
        for j in range(num_simulations):
            (line,) = ax.plot([], [], lw=1.5)
            lines.append(line)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for j, line in enumerate(lines):
                line.set_data(time_np[:frame], X_np[j, :frame])
            return lines

        ani = FuncAnimation(
            fig,
            update,
            frames=self.steps,
            init_func=init,
            blit=False,
            interval=interval,
        )

        writer = FFMpegWriter(fps=10)   # fps = frames per second
        ani.save("sde.gif", writer=writer)

        plt.show()

class SingleDimGravitationalPotential(LangevinDynamicsSimulator):
    def __init__(self,m, x0, t0, K_b, T, lambda_, device,delta_t, g=None):
        super().__init__(m, x0, t0, K_b, T, lambda_, device,delta_t=delta_t)
        if g == None:
            self.g = 9.81

    def run_overdamped_simulations(self, num_simulations=0, steps=0):
        assert num_simulations > 0, "Number of simulations must be > 0"
        assert type(num_simulations) == int, "Number of simulations must be an integer"
        assert steps > 0, "Number of steps must be > 0"
        assert type(steps) == int, "Number of steps must be an integer"


        self.initialize_simulation(num_simulations, steps)

        std = torch.sqrt(torch.tensor(2 * self.D * self.delta_t)).to(self.device)

        for i in range(steps): 
            self.all_positions[:, i] = self.x 
            self.all_times[i] = self.t

            noise = torch.normal(mean=0.0, std=std, size=(num_simulations,), device=self.device)

            self.x += (-self.m / self.lambda_) * self.g * self.delta_t + noise

            self.t += self.delta_t

        #print(self.all_positions)

        return (torch.max(self.all_positions), 
                torch.min(self.all_positions),
                torch.max(self.all_times), 
                torch.max(self.all_times)
            )

    def run_underdamped_simulations(self, num_simulations, steps, v0):
        assert num_simulations > 0, "Number of simulations must be > 0"
        assert type(num_simulations) == int, "Number of simulations must be an integer"
        assert steps > 0, "Number of steps must be > 0"
        assert type(steps) == int, "Number of steps must be an integer"


        self.initialize_simulation(num_simulations, steps)

        std = torch.sqrt(torch.tensor((2 * self.lambda_ * self.K_b * self.T * self.delta_t)/(self.m **2) )).to(self.device)

        v = v0
        all_velocities = torch.zeros((num_simulations, steps), device=self.device)

        for i in range(steps): 
            self.all_positions[:, i] = self.x 
            all_velocities[:, i] = v
            self.all_times[i] = self.t

            noise = torch.normal(mean=0.0, std=std, size=(num_simulations,), device=self.device)

            v += (-self.lambda_ * v - self.g * self.m + torch.exp((self.m * self.g * self.x)/(self.K_b * self.T))) / self.m * self.delta_t + noise
            self.x += v * self.delta_t

            self.t += self.delta_t

        #print(self.all_positions)

        return (torch.max(self.all_positions), 
                torch.min(self.all_positions),
                torch.max(self.all_times), 
                torch.max(self.all_times))
    
