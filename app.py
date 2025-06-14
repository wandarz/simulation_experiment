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
    masses = data.get('masses', [5, 5])
    velocities = data.get('velocities', [1, -1])
    m1, m2 = float(masses[0]), float(masses[1])
    v1, v2 = float(velocities[0]), float(velocities[1])
    # Initial positions (opposite sides, center of mass at origin)
    r = 100
    x1_0, y1_0 = -r * m2 / (m1 + m2), 0
    x2_0, y2_0 = r * m1 / (m1 + m2), 0
    # Initial velocities (perpendicular, for circular orbits if possible)
    vx1_0, vy1_0 = 0, v1
    vx2_0, vy2_0 = 0, v2
    # State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    y0 = [x1_0, y1_0, x2_0, y2_0, vx1_0, vy1_0, vx2_0, vy2_0]
    G = 1.0
    def deriv(t, y):
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
        dx = x2 - x1
        dy = y2 - y1
        dist3 = (dx**2 + dy**2)**1.5 + 1e-8
        fx = G * m1 * m2 * dx / dist3
        fy = G * m1 * m2 * dy / dist3
        return [vx1, vy1, vx2, vy2,
                fx / m1, fy / m1, -fx / m2, -fy / m2]
    t_span = (0, 20)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval, rtol=1e-8)
    x1, y1, x2, y2 = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    vx1, vy1, vx2, vy2 = sol.y[4], sol.y[5], sol.y[6], sol.y[7]
    return jsonify({
        't': t_eval.tolist(),
        'x1': x1.tolist(), 'y1': y1.tolist(),
        'x2': x2.tolist(), 'y2': y2.tolist(),
        'vx1': vx1.tolist(), 'vy1': vy1.tolist(),
        'vx2': vx2.tolist(), 'vy2': vy2.tolist(),
    })

if __name__ == '__main__':
    app.run(debug=True) 