import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# Constants
LENGTH = 5  # Length of the heat exchanger (m)
NUM_POINTS = 200  # Number of discretization points

DIAMETER_INNER = 0.02  # Inner tube diameter (m)
THICKNESS_INNER = 0.001  # Inner tube thickness (m)

DIAMETER_OUTER = 0.05  # Outer tube diameter (m)
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

fluid1 = 'Water'
mdot1 = 0.1  # Mass flow rate (kg/s)
P1 = 101325  # Pressure (Pa)
H1 = 440000  # Enthalpy (J/kg)


fluid2 = 'Water'
mdot2 = 0.1  # Mass flow rate (kg/s)
P2 = 101325  # Pressure (Pa)
H2 = 80000  # Enthalpy (J/kg)

A = np.diag(-1*np.ones(NUM_POINTS-1, dtype=float), -1) + np.diag(np.ones(NUM_POINTS-1, dtype=float), 1)
A[0,0] = 1
A[0,1] = 0
A[NUM_POINTS-1,NUM_POINTS-1] = 1

dx = LENGTH/(NUM_POINTS-1)

h1 = np.ones(NUM_POINTS, dtype=float) * H1
h2 = np.ones(NUM_POINTS, dtype=float) * H2

b1 = np.zeros_like(h1)
b2 = np.zeros_like(h2)

b1[0] = h1[0]
b2[0] = h2[0]

delta = 1e6
for j in range(50):
    h1_old = h1.copy()
    h2_old = h2.copy()
    for i in range(1,NUM_POINTS):
        state1 = state(fluid1,P1,h1[i])
        state2 = state(fluid2,P2,h2[i])
        v1 = mdot1/state1['D']/A1
        v2 = mdot2/state2['D']/A2
        q_ = q(state1,state2,mdot1,mdot2)
        
        b1[i] = -2 * q_ * dx / mdot1
        b2[i] = 2 * q_ * dx / mdot2
        
        b1[-1] = b1[-1]/2
        b2[-1] = b2[-1]/2
        
    h1 = np.linalg.solve(A,b1)
    h2 = np.linalg.solve(A,b2)
    delta = np.max(np.abs(h1-h1_old)/h1_old) + np.max(np.abs(h2-h2_old)/h2_old)
    print(j,':',delta)
    if delta < 1e-6:
        break
    
    

x = np.linspace(0, LENGTH, NUM_POINTS)

# Calculate temperature and heat transfer rate along the length
T1 = [state(fluid1, P1, h)['T'] for h in h1]
T2 = [state(fluid2, P2, h)['T'] for h in h2]
q_values = [q(state(fluid1, P1, h1[i]), state(fluid2, P2, h2[i]), mdot1, mdot2) for i in range(NUM_POINTS)]

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot temperature vs x
axs[0].plot(x, T1, label='Inner Tube Temperature (T1)')
axs[0].plot(x, T2, label='Outer Tube Temperature (T2)')
axs[0].set_xlabel('Length (m)')
axs[0].set_ylabel('Temperature (K)')
axs[0].set_title('Temperature Distribution Along the Heat Exchanger')
axs[0].legend()
axs[0].grid()

# Plot heat transfer rate vs x
axs[1].plot(x, q_values, label='Heat Transfer Rate (q)')
axs[1].set_xlabel('Length (m)')
axs[1].set_ylabel('Heat Transfer Rate (W)')
axs[1].set_title('Heat Transfer Rate Along the Heat Exchanger')
axs[1].legend()
axs[1].grid()

# Plot enthalpy vs x
axs[2].plot(x, h1, label='Inner Tube Enthalpy (h1)')
axs[2].plot(x, h2, label='Outer Tube Enthalpy (h2)')
axs[2].set_xlabel('Length (m)')
axs[2].set_ylabel('Enthalpy (J/kg)')
axs[2].set_title('Enthalpy Distribution Along the Heat Exchanger')
axs[2].legend()
axs[2].grid()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()