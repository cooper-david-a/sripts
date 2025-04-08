import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# Constants
LENGTH = 1  # Length of the heat exchanger (m)
NUM_POINTS = 100*LENGTH  # Number of discretization points

DIAMETER_INNER = 0.02  # Inner tube diameter (m)
THICKNESS_INNER = 0.001  # Inner tube thickness (m)

DIAMETER_OUTER = 0.05  # Outer tube diameter (m)
THICKNESS_OUTER = 0.002  # Outer tube thickness (m)

K_TUBE = 24  # Thermal conductivity of the inner tube (W/m-K)
ROUGHNESS = 0  # Tube roughness (m)

D1 = DIAMETER_INNER - 2*THICKNESS_INNER  # Inner tube diameter (m)
D2 = DIAMETER_INNER
D3 = DIAMETER_OUTER - 2*THICKNESS_OUTER  # Outer tube diameter (m)

A1 = np.pi/4*D1**2  # Inner tube area (m^2)
A2 = np.pi/4*(D3**2-D2**2)  # Outer tube area (m^2)


def state(fluid, P, H):
    phase = CP.PhaseSI('P', P, 'H', H, fluid)
    return {'fluid': fluid, 'P': P, 'H': H,
            'Q': CP.PropsSI('Q', f'P|{phase}', P, 'H', H, fluid),
            'T': CP.PropsSI('T', f'P|{phase}', P, 'H', H, fluid),
            'D': CP.PropsSI('D', f'P|{phase}', P, 'H', H, fluid),
            'Pr': CP.PropsSI('Prandtl', f'P|{phase}', P, 'H', H, fluid),
            'mu': CP.PropsSI('viscosity', f'P|{phase}', P, 'H', H, fluid),
            'k': CP.PropsSI('conductivity', f'P|{phase}', P, 'H', H, fluid),
            }


def q(state1, state2, mdot1, mdot2):
    Dh2 = D3 - D2
    Re1 = mdot1*D1/A1/state1['mu']
    Re2 = mdot2*Dh2/A2/state2['mu']
    Pr1 = state1['Pr']
    Pr2 = state2['Pr']
    Q1 = state1['Q']
    Q2 = state2['Q']

    f1 = fsolve(lambda f: 1/np.sqrt(f) + 2*np.log10(ROUGHNESS /
                (3.7*D1)+2.51/(Re1*np.sqrt(f))), 0.02)[0]
    f2 = fsolve(lambda f: 1/np.sqrt(f) + 2*np.log10(ROUGHNESS /
                (3.7*Dh2)+2.51/(Re2*np.sqrt(f))), 0.02)[0]

    if 0 <= Q1 <= 1:
        Nu1 = 5000
    else:
        Nu1 = (f1/8)*(Re1-1000)*Pr1/(1+12.7*(f1/8)**0.5*(Pr1**(2/3)-1))

    if 0 <= Q2 <= 1:
        Nu2 = 5000
    else:
        Nu2 = (f2/8)*(Re2-1000)*Pr2/(1+12.7*(f2/8)**0.5*(Pr2**(2/3)-1))

    h1 = Nu1 * state1['k']/D1
    h2 = Nu2 * state2['k']/Dh2

    return np.pi*(state1['T']-state2['T']) / (1/(h1*D1) + np.log(D2/D1)/(2*K_TUBE) + 1/(h2*Dh2))


fluid1 = 'R32'
mdot1 = 0.015  # Mass flow rate (kg/s)
P1 = 20*101325  # Pressure (Pa)
H1 = 480000  # Enthalpy (J/kg)


fluid2 = 'Water'
mdot2 = 0.5  # Mass flow rate (kg/s)
P2 = 101325  # Pressure (Pa)
H2 = 50000  # Enthalpy (J/kg)

A = np.diag(-1.0*np.ones(NUM_POINTS-1, dtype=float), -1) + \
    np.diag(np.ones(NUM_POINTS-1, dtype=float), 1)
A[0, 0] = 1
A[0, 1] = 0
A[NUM_POINTS-1, NUM_POINTS-1] = 1

A = csr_matrix(A)

dx = LENGTH/(NUM_POINTS-1)
print('dx: ', dx)

h1 = np.ones(NUM_POINTS, dtype=float) * float(H1)
h2 = np.ones(NUM_POINTS, dtype=float) * float(H2)

b1 = np.zeros_like(h1)
b2 = np.zeros_like(h2)

b1[0] = h1[0]
b2[0] = h2[0]

for j in range(50):
    h1_old = h1.copy()
    h2_old = h2.copy()
    print('Iteration: ', j, 'Building b''s...')
    for i in range(1, NUM_POINTS):
        state1 = state(fluid1, P1, h1[i])
        state2 = state(fluid2, P2, h2[i])
        v1 = mdot1/state1['D']/A1
        v2 = mdot2/state2['D']/A2
        q_ = q(state1, state2, mdot1, mdot2)

        b1[i] = -2 * q_ * dx / mdot1
        b2[i] = 2 * q_ * dx / mdot2

        b1[-1] = b1[-1]/2
        b2[-1] = b2[-1]/2

    print('Iteration: ', j, 'Solving...')
    h1 = spsolve(A, b1)
    h2 = spsolve(A, b2)
    # Use a weighted average for faster convergence
    relaxation_factor = 0.5
    h1 = relaxation_factor * h1 + (1 - relaxation_factor) * h1_old
    h2 = relaxation_factor * h2 + (1 - relaxation_factor) * h2_old
    
    delta = abs(np.max(h1-h1_old)/np.max(h1_old)) + \
        abs(np.max(h2-h2_old)/np.max(h2_old))

    print(j, ':','delta: ', delta)
    if delta < 1e-6:
        break


x = np.linspace(0, LENGTH, NUM_POINTS)

# Calculate temperature and heat transfer rate along the length
T1 = [state(fluid1, P1, h)['T'] for h in h1]
T2 = [state(fluid2, P2, h)['T'] for h in h2]
q_values = [q(state(fluid1, P1, h1[i]), state(fluid2, P2, h2[i]),
              mdot1, mdot2) for i in range(NUM_POINTS)]

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
axs[1].set_ylabel('Heat Transfer Rate (W/m)')
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

# Calculate vapor quality, Reynolds number, and fluid velocity along the length
Q1 = [state(fluid1, P1, h)['Q'] for h in h1]
Q2 = [state(fluid2, P2, h)['Q'] for h in h2]
Re1 = [mdot1 * D1 / A1 / state(fluid1, P1, h)['mu'] for h in h1]
Re2 = [mdot2 * (D3 - D2) / A2 / state(fluid2, P2, h)['mu'] for h in h2]
v1 = [mdot1 / state(fluid1, P1, h)['D'] / A1 for h in h1]
v2 = [mdot2 / state(fluid2, P2, h)['D'] / A2 for h in h2]

# Create subplots for vapor quality, Reynolds number, and fluid velocity
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot vapor quality vs x
axs[0].plot(x, Q1, label='Inner Tube Vapor Quality (Q1)')
axs[0].plot(x, Q2, label='Outer Tube Vapor Quality (Q2)')
axs[0].set_xlabel('Length (m)')
axs[0].set_ylabel('Vapor Quality')
axs[0].set_title('Vapor Quality Distribution Along the Heat Exchanger')
axs[0].legend()
axs[0].grid()

# Plot Reynolds number vs x
axs[1].plot(x, Re1, label='Inner Tube Reynolds Number (Re1)')
axs[1].plot(x, Re2, label='Outer Tube Reynolds Number (Re2)')
axs[1].set_xlabel('Length (m)')
axs[1].set_ylabel('Reynolds Number')
axs[1].set_title('Reynolds Number Distribution Along the Heat Exchanger')
axs[1].legend()
axs[1].grid()

# Plot fluid velocity vs x
axs[2].plot(x, v1, label='Inner Tube Velocity (v1)')
axs[2].plot(x, v2, label='Outer Tube Velocity (v2)')
axs[2].set_xlabel('Length (m)')
axs[2].set_ylabel('Velocity (m/s)')
axs[2].set_title('Fluid Velocity Distribution Along the Heat Exchanger')
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()
