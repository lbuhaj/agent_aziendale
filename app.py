from flask import Flask, render_template, request, jsonify
from agents import HumanCapitalSystem
from database import ingest_cvs
import os

app = Flask(__name__)
# Inizializziamo il sistema (assicurati che la chiave Groq sia settata prima)
system = HumanCapitalSystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sync', methods=['POST'])
def sync_data():
    message = ingest_cvs()
    return jsonify({"message": message})

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("text")
    answer = system.orchestrate(user_input)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)