from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)

CORS(app)

users = {}

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if email in users and users[email] == password:
        return jsonify({'message': 'Login successful!'})
    else:
        return jsonify({'message': 'Login failed. Invalid username or password.'}), 401
    

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if email in users:
        return jsonify({'message': 'Username already exists. Please choose a different one.'}), 400

    users[email] = password
    return jsonify({'message': 'Sign-up successful!'}), 201

if __name__ == '__main__':
    app.run(debug=True)