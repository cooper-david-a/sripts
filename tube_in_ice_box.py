import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# Constants
LENGTH = 2  # Length of the heat exchanger (m)

DIAMETER = 0.0127  # Inner tube diameter (m)
THICKNESS = 0.001  # Inner tube thickness (m)

K_TUBE = 24  # Thermal conductivity of the inner tube (W/m-K)
ROUGHNESS = 0  # Tube roughness (m)

D1 = DIAMETER - 2*THICKNESS  # Inner tube diameter (m)
D2 = DIAMETER

A1 = np.pi/4*D1**2  # Inner tube area (m^2)

T_OUTER = 273.15 # Outer tube temperature (K)

fluid1 = 'Nitrogen'
P1 = 5*101325  # Pressure (Pa)
T1 = 650 # Temperature (K)

def state(fluid, P, T):
    phase = 'gas'  
    
    state_dict = {
        'fluid': fluid,
        'P': P,
        'T': T,
        'H': CP.PropsSI('H', 'P', P1, 'T', T, fluid),
        'D': CP.PropsSI('D', f'P|{phase}', P, 'T', T, fluid),
        'Pr': max(0.001,CP.PropsSI('Prandtl', f'P|{phase}', P, 'T', T, fluid)),
        'mu': CP.PropsSI('viscosity', f'P|{phase}', P, 'T', T, fluid),
        'k': CP.PropsSI('conductivity', f'P|{phase}', P, 'T', T, fluid),
        'Cp': CP.PropsSI('C', f'P|{phase}', P, 'T', T, fluid)
        }

    return state_dict

def q(state1, mdot1):
    Re1 = mdot1*D1/A1/state1['mu']
    Pr1 = state1['Pr']

    f1 = fsolve(lambda f: 1/np.sqrt(f) + 2*np.log10(ROUGHNESS /
                (3.7*D1)+2.51/(Re1*np.sqrt(f))), 0.02)[0]

    Nu1 = (f1/8)*(Re1-1000)*Pr1/(1+12.7*(f1/8)**0.5*(Pr1**(2/3)-1))
    
    h1 = Nu1 * state1['k']/D1

    return np.pi*(state1['T']-T_OUTER) / (1/(h1*D1) + np.log(D2/D1)/(2*K_TUBE))

def evaluate_flow_case(mdot):
    Tavg_guess = (T1 + T_OUTER)/2
    state1 = state(fluid1, P1, Tavg_guess)
    Re1 = mdot*D1/A1/state1['mu']
    Pr1 = state1['Pr']

    f1 = fsolve(lambda f: 1/np.sqrt(f) + 2*np.log10(ROUGHNESS /
                (3.7*D1)+2.51/(Re1*np.sqrt(f))), 0.02)[0]

    Nu1 = (f1/8)*(Re1-1000)*Pr1/(1+12.7*(f1/8)**0.5*(Pr1**(2/3)-1)) if Re1 > 3000 else 3.66

    h1 = Nu1 * state1['k']/D1

    To_calc = T_OUTER + (T1-T_OUTER) * np.exp(-np.pi*D1*LENGTH*h1/(mdot*state1['Cp']))
    Tavg_calc = (T1 + To_calc)/2
    Tavg_error = Tavg_calc - Tavg_guess
    return state1, Nu1, To_calc, Tavg_error


mdot_array = np.logspace(-4, -2, 200)  # Mass flow rate (kg/s)
Nu_array = [evaluate_flow_case(mdot)[1] for mdot in mdot_array]
outlet_temps = [evaluate_flow_case(mdot)[2] - 273.15 for mdot in mdot_array]
tavg_errors = [evaluate_flow_case(mdot)[3] for mdot in mdot_array]


fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
axes[0].plot(mdot_array, outlet_temps, color='tab:blue')
axes[0].set_xscale('log')
axes[0].set_ylabel('Outlet temperature (°C)')
axes[0].set_title('Outlet Temperature vs Mass Flow Rate')
axes[0].grid(True, alpha=0.3)

axes[1].plot(mdot_array, tavg_errors, color='tab:orange')
axes[1].set_xscale('log')
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_xlabel('Mass flow rate (kg/s)')
axes[1].set_ylabel('Tavg error (K)')
axes[1].set_title('Average Temperature Error vs Mass Flow Rate')
axes[1].grid(True, alpha=0.3)

fig.tight_layout()

fig_re, ax_re = plt.subplots(figsize=(8, 4))
ax_re.plot(mdot_array, Nu_array, color='tab:green')
ax_re.set_xscale('log')
ax_re.set_xlabel('Mass flow rate (kg/s)')
ax_re.set_ylabel('Nusselt number')
ax_re.set_title('Nusselt Number vs Mass Flow Rate')
ax_re.grid(True, alpha=0.3)

plt.show()