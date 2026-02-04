from flask import Flask, render_template, request, jsonify
from agents import HumanCapitalSystem
from database import ingest_cvs
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import os

# Carica il file .env all'avvio
load_dotenv()

app = Flask(__name__)
system = HumanCapitalSystem()

# Carica le configurazioni
UPLOAD_FOLDER = './data/cv_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

system = HumanCapitalSystem()

try:
    system = HumanCapitalSystem()
except Exception as e:
    print(f"Critico: Impossibile avviare HumanCapitalSystem: {e}")
    system = None

# Analisi automatica all'avvio
with app.app_context():
    try:
        print("🔍 Analisi CV all'avvio...")
        ingest_cvs()
        print("✅ Analisi completata.")
    except Exception as e:
        print(f"⚠️ Errore durante l'ingest automatico: {e}")



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sync', methods=['POST'])
def sync_data():
    message = ingest_cvs()
    return jsonify({"message": message})

@app.route('/ask', methods=['POST'])
def ask():
    if not system:
        return jsonify({"answer": "Sistema non configurato correttamente. Controlla il terminale."})
    
    try:
        data = request.get_json()
        user_input = data.get("text", "")
        print(f"📩 Ricevuta domanda: {user_input}")
        
        answer = system.orchestrate(user_input)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"🔥 Errore durante l'elaborazione della domanda: {e}")
        return jsonify({"answer": "Si è verificato un errore interno."}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "Nessun file inviato"}), 400
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            ingest_cvs()
            return jsonify({"message": f"CV {filename} analizzato con successo!"})
        return jsonify({"message": "Formato file non valido."}), 400
    except Exception as e:
        return jsonify({"message": f"Errore upload: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)