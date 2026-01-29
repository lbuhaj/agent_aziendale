import os
import ollama
from groq import Groq
from database import get_vector_db

class HumanCapitalSystem:
    def __init__(self):
        self.db = get_vector_db()
        # Assicurati di avere GROQ_API_KEY nel tuo ambiente
        self.client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # 1. AGENTE PARSER (Ollama - Locale)
    def parser_agent(self, raw_text):
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': 'Sei un esperto HR. Estrai le competenze tecniche da questo testo.'},
            {'role': 'user', 'content': raw_text}
        ])
        return response['message']['content']

    # 2. AGENTE RICERCA (ChromaDB)
    def search_agent(self, query):
        results = self.db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])

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
    def orchestrate(self, user_message):
        # Il Planner decide di cercare nel DB
        context = self.search_agent(user_message)
        if not context:
            return "Nessun profilo trovato nel database aziendale."
        
        # Il Planner chiede al Critic di validare e rispondere
        return self.critic_agent(context, user_message)