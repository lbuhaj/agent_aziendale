import os
import ollama
from groq import Groq
from database import get_vector_db
from dotenv import load_dotenv

# Carica le variabili dal file .env
load_dotenv(override=True)

class HumanCapitalSystem:
    def __init__(self):
        self.db = get_vector_db()
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key or api_key == "tua_chiave_groq_qui":
            print("❌ ERRORE: GROQ_API_KEY non trovata o non valida nel file .env")
            self.client_groq = None
        else:
            print("✅ Chiave Groq caricata correttamente.")
            self.client_groq = Groq(api_key=api_key)

    # 1. AGENTE PARSER (Ollama - Locale)
    def parser_agent(self, raw_text):
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': 'Sei un esperto HR. Estrai le competenze tecniche da questo testo.'},
            {'role': 'user', 'content': raw_text}
        ])
        return response['message']['content']

    # 2. AGENTE RICERCA (ChromaDB)
    def search_agent(self, query):
        try:
            # Cerchiamo nel DB basandoci sull'ultima query dell'utente
            results = self.db.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in results])
        except:
            return ""


    # 3. AGENTE CRITIC (Groq - Llama 3.1)
    def critic_agent(self, context, query):
        chat_completion = self.client_groq.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Sei un Critic HR. Valuta i profili trovati rispetto alla richiesta dell'utente."
                },
                {
                    "role": "user",
                    "content": f"Richiesta: {query}\n\nProfili dal DB:\n{context}",
                }
            ],
            model="llama-3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content

    # 4. AGENTE PLANNER (Orchestratore)
    # Modifica il metodo orchestrate in agents.py
    def orchestrate(self, user_message):
        if not self.client_groq:
            return "Errore di configurazione: Controlla la tua GROQ_API_KEY nel file .env"

        context = self.search_agent(user_message)
        
        try:
            chat_completion = self.client_groq.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Sei un assistente HR aziendale. Rispondi amichevolmente o cerca nei CV se richiesto."
                    },
                    {
                        "role": "user",
                        "content": f"CONTESTO: {context}\n\nDOMANDA: {user_message}",
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"❌ Errore API Groq: {e}")
            return f"Errore nell'invocazione di Groq: {str(e)}"
