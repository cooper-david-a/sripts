import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# Constants
LENGTH = 10.0  # Length of the heat exchanger (m)

DIAMETER_INNER = 0.02  # Inner tube diameter (m)
THICKNESS_INNER = 0.001  # Inner tube thickness (m)

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
    P1 = 101325  # Pressure of fluid 1 (Pa)
    P2 = 101325  # Pressure of fluid 2 (Pa)
    state1 = state('Water', P1, 80000)
    state2 = state('Water', P2, 300000)
    mdot1 = 0.5  # Mass flow rate of fluid 1 (kg/s)
    mdot2 = 0.5  # Mass flow rate of fluid 2 (kg/s)
    
    def dh_dx(x, h2):
        state2 = state('Water',P2 , h2[0])
        qx = q(state1, state2, mdot1, mdot2)
        v = mdot2/state2['D']/A2
        return qx/v
    
    # Define the range of the heat exchanger
    x_span = (0, LENGTH)
    NUM_POINTS = 100
    x_eval = np.linspace(x_span[0], x_span[1], NUM_POINTS)

    # Initial enthalpy for the outer tube fluid
    h2_initial = state2['H']

    # Solve the ODE using RK45
    solution = solve_ivp(dh_dx, x_span, [h2_initial], t_eval=x_eval, method='RK45')

    # Extract results
    x = solution.t
    h2_values = solution.y[0]

    # Calculate corresponding temperatures for the outer tube fluid
    T2_values = [state('Water', state2['P'], h)['T'] for h in h2_values]

    # Assuming constant temperature for the inner tube fluid
    T1_values = [state1['T']] * len(x)

    # Calculate heat transfer rate (q) along the heat exchanger
    q_values = [q(state1, state('Water', state2['P'], h), 0.5, 0.5) for h in h2_values]

    # Plot T2 vs x
    plt.figure(figsize=(8, 6))
    plt.plot(x, T2_values, label='T2 (Outer Tube Fluid)', color='blue')
    plt.xlabel('Position along the Heat Exchanger (x) [m]')
    plt.ylabel('Temperature (T) [K]')
    plt.title('Temperature Profile of Outer Tube Fluid')
    plt.legend()
    plt.grid()
    plt.show()