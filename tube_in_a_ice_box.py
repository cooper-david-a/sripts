import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# Constants
LENGTH = 1  # Length of the heat exchanger (m)
NUM_POINTS = 100*LENGTH  # Number of discretization points

DIAMETER = 0.02  # Inner tube diameter (m)
THICKNESS = 0.001  # Inner tube thickness (m)

K_TUBE = 24  # Thermal conductivity of the inner tube (W/m-K)
ROUGHNESS = 0  # Tube roughness (m)

D1 = DIAMETER - 2*THICKNESS  # Inner tube diameter (m)
D2 = DIAMETER

A1 = np.pi/4*D1**2  # Inner tube area (m^2)

T_OUTER = 273 # Outer tube temperature (K)

fluid1 = 'Nitrogen'
mdot1 = 0.01  # Mass flow rate (kg/s)
P1 = 2*101325  # Pressure (Pa)
T1 = 650 # Temperature (K)
H1 = CP.PropsSI('H', 'P', P1, 'T', T1, fluid1)  # Enthalpy (J/kg)


def state(fluid, P, H):
    phase = CP.PhaseSI('P', P, 'H', H, fluid)
    
    if phase == "twophase":
        Q = CP.PropsSI('Q', f'P|{phase}', P, 'H', H, fluid)
        k = Q*CP.PropsSI('conductivity', f'P|{phase}', P, 'Q', 1, fluid) + (1-Q)*CP.PropsSI('conductivity', f'P|{phase}', P, 'Q', 0, fluid)
    else:
        Q = -1
        k = CP.PropsSI('conductivity', f'P|{phase}', P, 'H', H, fluid)
    
    return {'fluid': fluid, 'P': P, 'H': H,
            'Q': Q,
            'T': CP.PropsSI('T', f'P|{phase}', P, 'H', H, fluid),
            'D': CP.PropsSI('D', f'P|{phase}', P, 'H', H, fluid),
            'Pr': CP.PropsSI('Prandtl', f'P|{phase}', P, 'H', H, fluid),
            'mu': CP.PropsSI('viscosity', f'P|{phase}', P, 'H', H, fluid),
            'k': k
            }


def q(state1, mdot1):
    Re1 = mdot1*D1/A1/state1['mu']
    Pr1 = state1['Pr']
    Q1 = state1['Q']

    f1 = fsolve(lambda f: 1/np.sqrt(f) + 2*np.log10(ROUGHNESS /
                (3.7*D1)+2.51/(Re1*np.sqrt(f))), 0.02)[0]

    if 0 <= Q1 <= 1:
        Nu1 = 5000
    else:
        Nu1 = (f1/8)*(Re1-1000)*Pr1/(1+12.7*(f1/8)**0.5*(Pr1**(2/3)-1))

    h1 = Nu1 * state1['k']/D1

    return np.pi*(state1['T']-T_OUTER) / (1/(h1*D1) + np.log(D2/D1)/(2*K_TUBE))

A = np.diag(-1.0*np.ones(NUM_POINTS-1, dtype=float), -1) + \
    np.diag(np.ones(NUM_POINTS-1, dtype=float), 1)
A[0, 0] = 1
A[0, 1] = 0
A[NUM_POINTS-1, NUM_POINTS-1] = 1

A = csr_matrix(A)

dx = LENGTH/(NUM_POINTS-1)
print('dx: ', dx)


def solve_enthalpy_profile(mdot_value, initial_guess=None):
    h = np.ones(NUM_POINTS, dtype=float) * float(H1)
    if initial_guess is not None:
        h = initial_guess.copy()

    for j in range(50):
        h_old = h.copy()
        print('Iteration: ', j, 'Building b''s...')
        b = np.zeros_like(h)
        b[0] = h[0]

        for i in range(1, NUM_POINTS):
            state1 = state(fluid1, P1, h[i])
            q_ = q(state1, mdot_value)
            b[i] = -2 * q_ * dx / mdot_value

        b[-1] = b[-1] / 2

        print('Iteration: ', j, 'Solving...')
        h = spsolve(A, b)

        relaxation_factor = 1.0
        h = relaxation_factor * h + (1 - relaxation_factor) * h_old

        delta = np.max(np.abs(h - h_old)) / max(np.max(np.abs(h_old)), 1e-30)

        print(j, ':', 'delta: ', delta)
        if delta < 1e-6:
            break

    return h


base_profile = solve_enthalpy_profile(mdot1)

x = np.linspace(0, LENGTH, NUM_POINTS)

# Calculate temperature and heat transfer rate along the length
T1 = [state(fluid1, P1, h)['T'] for h in base_profile]
q_values = [q(state(fluid1, P1, base_profile[i]), mdot1) for i in range(NUM_POINTS)]

outlet_state = state(fluid1, P1, base_profile[-1])
exit_temperature = outlet_state['T']
total_heat_transfer = mdot1 * (base_profile[-1] - base_profile[0])

print(f'Exit temperature: {exit_temperature:.2f} K')
print(f'Total heat transfer: {total_heat_transfer:.2f} W')

# Create subplots for the base case profile
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

axs[0].plot(x, T1, label='Inner Tube Temperature (T1)')
axs[0].set_xlabel('Length (m)')
axs[0].set_ylabel('Temperature (K)')
axs[0].set_title('Temperature Distribution Along the Heat Exchanger')
axs[0].legend()
axs[0].grid()

axs[1].plot(x, q_values, label='Heat Transfer Rate (q)')
axs[1].set_xlabel('Length (m)')
axs[1].set_ylabel('Heat Transfer Rate (W/m)')
axs[1].set_title('Heat Transfer Rate Along the Heat Exchanger')
axs[1].legend()
axs[1].grid()

axs[2].plot(x, base_profile, label='Inner Tube Enthalpy (h1)')
axs[2].set_xlabel('Length (m)')
axs[2].set_ylabel('Enthalpy (J/kg)')
axs[2].set_title('Enthalpy Distribution Along the Heat Exchanger')
axs[2].legend()
axs[2].grid()

plt.tight_layout()

# Sweep over mass flow rate
mdot_values = np.logspace(np.log10(0.1 * mdot1), np.log10(10 * mdot1), 20)
exit_temperatures = []
total_heat_transfers = []
previous_profile = None

for mdot_value in mdot_values:
    profile = solve_enthalpy_profile(mdot_value, initial_guess=previous_profile)
    outlet_state = state(fluid1, P1, profile[-1])
    exit_temperatures.append(outlet_state['T'])
    total_heat_transfers.append(mdot_value * (profile[-1] - profile[0]))
    previous_profile = profile

fig_sweep, axs_sweep = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

axs_sweep[0].plot(mdot_values, exit_temperatures, marker='o', label='Exit temperature')
axs_sweep[0].set_ylabel('Exit Temperature (K)')
axs_sweep[0].set_title('Sweep over Mass Flow Rate')
axs_sweep[0].grid(True)
axs_sweep[0].legend()

axs_sweep[1].plot(mdot_values, total_heat_transfers, marker='o', color='C1', label='Total heat transfer')
axs_sweep[1].set_xlabel('Mass flow rate (kg/s)')
axs_sweep[1].set_ylabel('Total Heat Transfer (W)')
axs_sweep[1].grid(True)
axs_sweep[1].legend()

plt.tight_layout()
plt.show()
