from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
from chatbot import get_response  # Import get_response function from chatbot.py

app = Flask(__name__)
CORS(app)  # Apply CORS to your Flask app

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data['query']
    response = get_response(user_query)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
