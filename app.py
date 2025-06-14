from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    # Here you could process the simulation with the given masses, but for now just echo back
    return jsonify({'status': 'ok', 'masses': data.get('masses', [1, 1])})

if __name__ == '__main__':
    app.run(debug=True) 