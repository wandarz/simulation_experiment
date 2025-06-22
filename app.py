from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.integrate import solve_ivp

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    planets = data.get('planets', [
        {'mass': 5, 'velocity': 1, 'velocity_angle': 90, 'color': '#1E90FF'},
        {'mass': 5, 'velocity': 1, 'velocity_angle': 270, 'color': '#FFD700'}
    ])
    time_interval = float(data.get('time_interval', 20))
    
    n_planets = len(planets)
    if n_planets < 1:
        return jsonify({'error': 'At least one planet is required'}), 400
    
    # Extract planet properties
    masses = [float(p['mass']) for p in planets]
    velocities = [float(p['velocity']) for p in planets]
    velocity_angles = [np.deg2rad(float(p['velocity_angle'])) for p in planets]
    colors = [p.get('color', '#FFFFFF') for p in planets]
    
    # Calculate initial positions (arrange in a circle)
    r = 100
    positions = []
    for i in range(n_planets):
        if i == 0:
            # First planet at origin
            positions.append([0, 0])
        else:
            # Other planets positioned based on their initial_distance from planet 1
            initial_distance = float(planets[i].get('initial_distance', 100))
            angle = 2 * np.pi * i / (n_planets - 1) if n_planets > 1 else 0
            x = initial_distance * np.cos(angle)
            y = initial_distance * np.sin(angle)
            positions.append([x, y])
    
    # Calculate center of mass and adjust positions
    total_mass = sum(masses)
    cm_x = sum(m * pos[0] for m, pos in zip(masses, positions)) / total_mass
    cm_y = sum(m * pos[1] for m, pos in zip(masses, positions)) / total_mass
    
    # Adjust positions so center of mass is at origin
    for pos in positions:
        pos[0] -= cm_x
        pos[1] -= cm_y
    
    # Initial velocities from magnitude and angle
    initial_velocities = []
    for v, angle in zip(velocities, velocity_angles):
        vx = v * np.cos(angle)
        vy = v * np.sin(angle)
        initial_velocities.append([vx, vy])
    
    # State vector: [x1, y1, x2, y2, ..., vx1, vy1, vx2, vy2, ...]
    y0 = []
    for pos in positions:
        y0.extend(pos)
    for vel in initial_velocities:
        y0.extend(vel)
    
    G = 1.0
    
    def deriv(t, y):
        # Extract positions and velocities
        positions = []
        velocities = []
        for i in range(n_planets):
            x = y[2*i]
            y_pos = y[2*i + 1]
            vx = y[2*n_planets + 2*i]
            vy = y[2*n_planets + 2*i + 1]
            positions.append([x, y_pos])
            velocities.append([vx, vy])
        
        # Calculate accelerations due to gravitational forces
        accelerations = [[0, 0] for _ in range(n_planets)]
        
        for i in range(n_planets):
            for j in range(n_planets):
                if i != j:
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist3 = (dx**2 + dy**2)**1.5 + 1e-8
                    
                    # Gravitational force
                    fx = G * masses[i] * masses[j] * dx / dist3
                    fy = G * masses[i] * masses[j] * dy / dist3
                    
                    # Acceleration = force / mass
                    accelerations[i][0] += fx / masses[i]
                    accelerations[i][1] += fy / masses[i]
        
        # Return derivatives: [vx1, vy1, vx2, vy2, ..., ax1, ay1, ax2, ay2, ...]
        result = []
        for vel in velocities:
            result.extend(vel)
        for acc in accelerations:
            result.extend(acc)
        
        return result
    
    t_span = (0, time_interval)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval, rtol=1e-8)
    
    # Extract results
    result = {
        't': t_eval.tolist(),
        'planets': []
    }
    
    for i in range(n_planets):
        planet_data = {
            'x': sol.y[2*i].tolist(),
            'y': sol.y[2*i + 1].tolist(),
            'vx': sol.y[2*n_planets + 2*i].tolist(),
            'vy': sol.y[2*n_planets + 2*i + 1].tolist(),
            'mass': masses[i],
            'color': colors[i]
        }
        result['planets'].append(planet_data)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 