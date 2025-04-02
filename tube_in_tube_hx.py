import numpy as np
from scipy.optimize import fsolve
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# Constants
LENGTH = 10.0  # Length of the heat exchanger (m)
DIAMETER_INNER = 0.02  # Inner tube diameter (m)
THICKNESS_INNER = 0.001  # Inner tube thickness (m)
NUM_POINTS = 10  # Number of points to discretize the heat exchanger

DIAMETER_OUTER = 0.04  # Outer tube diameter (m)
THICKNESS_OUTER = 0.002  # Outer tube thickness (m)

K_TUBE = 16  # Thermal conductivity of the inner tube (W/m-K)
ROUGHNESS = 0 # Tube roughness (m)

D1 = DIAMETER_INNER - 2*THICKNESS_INNER  # Inner tube diameter (m)
D2 = DIAMETER_INNER
D3 = DIAMETER_OUTER - 2*THICKNESS_OUTER  # Outer tube diameter (m)

A1 = np.pi/4*D1**2  # Inner tube area (m^2)
A2 = np.pi/4*(D3**2-D2**2)  # Outer tube area (m^2)

def state(fluid,P,H):
    return {'fluid':fluid,'P':P,'H':H,
            'T':CP.PropsSI('T','P',P,'H',H,fluid),
            'D':CP.PropsSI('D','P',P,'H',H,fluid),
            'Pr':CP.PropsSI('Prandtl','P',P,'H',H,fluid),
            'mu':CP.PropsSI('viscosity','P',P,'H',H,fluid),
            'k':CP.PropsSI('conductivity','P',P,'H',H,fluid)
            }
  
        
def q(state1, state2, mdot1, mdot2):
    Dh2 = D3 - D2
    Re1 = mdot1*D1/A1/state1['mu']
    Re2 = mdot2*Dh2/A2/state2['mu']
    Pr1 = state1['Pr']
    Pr2 = state2['Pr']
    
    f1 = fsolve(lambda f: 1/np.sqrt(f) + 2*np.log10(ROUGHNESS/(3.7*D1)+2.51/(Re1*np.sqrt(f))),0.02)[0]
    f2 = fsolve(lambda f: 1/np.sqrt(f) + 2*np.log10(ROUGHNESS/(3.7*Dh2)+2.51/(Re2*np.sqrt(f))),0.02)[0]
    
    Nu1 =  (f1/8)*(Re1-1000)*Pr1/(1+12.7*(f1/8)**0.5*(Pr1**(2/3)-1))
    Nu2 =  (f2/8)*(Re2-1000)*Pr2/(1+12.7*(f2/8)**0.5*(Pr2**(2/3)-1))
    
    h1 = Nu1 * state1['k']/D1
    h2 = Nu2 * state2['k']/Dh2
    
    return np.pi*(state1['T']-state2['T']) / (1/(h1*D1) + np.log(D2/D1)/(2*K_TUBE) + 1/(h2*Dh2))

if __name__ == '__main__':
    fluid1 = 'Water'
    fluid2 = 'Water'
    P1 = 100000
    P2 = 100000
    T1 = 350
    T2 = 300
    H1 = CP.PropsSI('H','T',T1,'P',P1,fluid1)
    H2 = CP.PropsSI('H','T',T2,'P',P2,fluid2)
    state1 = state(fluid1,P1,H1)
    state2 = state(fluid2,P2,H2)
    
    mdot1 = 0.1
    mdot2 = 0.1
    
    x = np.linspace(0,LENGTH,NUM_POINTS)
    dx = x[1]-x[0]
    
    A = np.diag(-1*np.ones(2*NUM_POINTS-1),-1) + np.diag(np.ones(2*NUM_POINTS-1),1)
    A[0,0] = 1
    A[0,1] = 0
    A[NUM_POINTS-1,NUM_POINTS] = 0
    A[NUM_POINTS,NUM_POINTS] = 1
    A[NUM_POINTS,NUM_POINTS-1] = 0
    A[NUM_POINTS,NUM_POINTS+1] = 0
    
    h = np.append(np.ones(NUM_POINTS)*H1,np.ones(NUM_POINTS)*H2)
    h.shape = (2*NUM_POINTS,1)
    b = np.zeros_like(h)
    b[0] = H1
    b[NUM_POINTS] = H2
    
    

    def equations(h):
        h.shape = (2*NUM_POINTS,1)
        for i in range(1,NUM_POINTS):
            state1 = state(fluid1,P1,h[i].item())
            state2 = state(fluid2,P2,h[NUM_POINTS+i].item())
            b[i] = 2*q(state1,state2,mdot1,mdot2)*dx
            b[NUM_POINTS+i] = q(state1,state2,mdot1,mdot2)*dx
        result = (A.dot(h) - b).flatten()
        return result
    
    h = fsolve(equations,h)

    T1_values = [CP.PropsSI('T', 'P', P1, 'H', h[i], fluid1) for i in range(NUM_POINTS)]
    T2_values = [CP.PropsSI('T', 'P', P2, 'H', h[NUM_POINTS + i], fluid2) for i in range(NUM_POINTS)]

    # Calculate heat transfer rate (q) values
    q_values = [
        q(state(fluid1, P1, h[i]), state(fluid2, P2, h[NUM_POINTS + i]), mdot1, mdot2)
        for i in range(NUM_POINTS)
    ]

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))

    # Plot temperature profiles
    axs[0].plot(x, T1_values, label='T1 (Inner Tube)')
    axs[0].plot(x, T2_values, label='T2 (Outer Tube)')
    axs[0].set_xlabel('Length (m)')
    axs[0].set_ylabel('Temperature (K)')
    axs[0].set_title('Temperature Profile Along the Heat Exchanger')
    axs[0].legend()
    axs[0].grid()

    # Plot heat transfer rate (q) profile
    axs[1].plot(x, q_values, label='Heat Transfer Rate (q)', color='red')
    axs[1].set_xlabel('Length (m)')
    axs[1].set_ylabel('Heat Transfer Rate (W)')
    axs[1].set_title('Heat Transfer Rate Along the Heat Exchanger')
    axs[1].legend()
    axs[1].grid()
    
        # Plot enthalpy profiles
    h1_values = [h[i] for i in range(NUM_POINTS)]  # Enthalpy for fluid 1
    h2_values = [h[NUM_POINTS + i] for i in range(NUM_POINTS)]  # Enthalpy for fluid 2
    axs[2].plot(x, h1_values, label='h1 (Inner Tube)', color='blue')
    axs[2].plot(x, h2_values, label='h2 (Outer Tube)', color='green')
    axs[2].set_xlabel('Length (m)')
    axs[2].set_ylabel('Enthalpy (J/kg)')
    axs[2].set_title('Enthalpy Profile Along the Heat Exchanger')
    axs[2].legend()
    axs[2].grid()

    # Show the plots
    plt.tight_layout()
    plt.show()