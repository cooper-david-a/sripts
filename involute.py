import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Define spline parameters
N = 30            # Number of teeth
m = 2.54            # Module
phi = np.radians(14.5)  # Pressure angle in radians

# Calculated values
Dp = N * m                   # Pitch diameter
Db = Dp * np.cos(phi)        # Base circle diameter
Rb = Db / 2                  # Base circle radius
Dr = Dp - 2.5 * m            # Root diameter (adjust based on standard)
DM = Dp + 1.25 * m           # Major diameter (for clearance)

# Generate involute curve
t_vals = np.linspace(0, np.sqrt((DM**2 - Db**2) / Db**2), 100)

x_vals = Rb * (np.cos(t_vals) + t_vals * np.sin(t_vals))
y_vals = Rb * (np.sin(t_vals) - t_vals * np.cos(t_vals))

# Rotate and duplicate teeth
theta_step = 2 * np.pi / N
points = []
for i in range(N):
    angle = i * theta_step
    x_rot = x_vals * np.cos(angle) - y_vals * np.sin(angle)
    y_rot = x_vals * np.sin(angle) + y_vals * np.cos(angle)    
    points.append((x_rot, y_rot))
    
for i in range(N):
    angle = theta_step * (i + .4)
    x_rot = x_vals * np.cos(angle) - y_vals * np.sin(angle)
    y_rot = x_vals * np.sin(angle) + y_vals * np.cos(angle)    
    points.append((x_rot, -y_rot))

# Plot the internal spline profile

for p in points:
    plt.plot(p[0], p[1], 'b-')

major_diameter_circle = plt.Circle((0,0),78.4098/2, color='r', fill=False )
minor_diameter_circle = plt.Circle((0,0),74.88555/2, color='g',fill=False) 
base_circle = plt.Circle((0,0),Db/2,color='k',fill=False)   
pitch_circle = plt.Circle((0,0),Dp/2,color='y',fill=False)   
plt.gca().set_aspect('equal')
plt.gca().add_patch(major_diameter_circle)
plt.gca().add_patch(minor_diameter_circle)
plt.gca().add_patch(base_circle)
plt.gca().add_patch(pitch_circle)
plt.title("Internal Involute Spline Profile")
print(Db,Dp)
plt.show()
