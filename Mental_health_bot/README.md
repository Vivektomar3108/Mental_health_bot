# 🧠 Mental Health Buddy Chatbot  

A warm, funny, and supportive **AI-powered mental health buddy** built using **Flask + LangChain + Groq + Chroma**.  
It listens to users, responds with empathy, humor, and expressiveness, and even speaks back using **Text-to-Speech**.  

---

## 🚀 Features  
- 🤝 Acts like a **caring friend** — empathetic, casual, and expressive.  
- 📚 Uses a **knowledge base** (`Peoject_2/Doc/`) to provide contextual responses.  
- 🔍 Retrieves relevant context via **Chroma vector database**.  
- 🧩 Memory-enabled chat (remembers conversation history).  
- 🗣️ Speaks responses aloud with **pyttsx3** (async TTS).  
- 🌐 Flask-based API with frontend (`index.html`).  

---

## 🛠️ Tech Stack  
- **Backend:** Flask  
- **LLM:** Groq (`gemma2-9b-it`) via LangChain  
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)  
- **Vector Store:** Chroma  
- **TTS:** pyttsx3  
- **Environment Config:** python-dotenv  

---

## 📂 Project Structure  
```
├── app.py                # Main Flask app with chatbot logic
├── templates/
│   └── index.html        # Frontend UI
├── Peoject_2/
│   └── Doc/              # Knowledge base documents
├── .env                  # API keys & environment variables
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Installation  

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/mental-health-buddy.git
   cd mental-health-buddy
   ```

2. **Create a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up `.env` file**  
   ```
   GROQ_API_KEY=your_api_key_here
   ```

---

## ▶️ Run the App  

```bash
python app.py
```

- Open in browser: **http://127.0.0.1:5000/**  
- Start chatting with your buddy 🤗  

---

## 📌 API Endpoints  

### `GET /`  
Loads the frontend.  

### `POST /chat`  
Send a chat message.  
**Request body:**  
```json
{
  "message": "Hey, I feel stressed today."
}
```  
**Response:**  
```json
{
  "response": "Aww, I hear you 💜. Stress can be tough — want me to share some relaxing tricks?"
}
```  

---

## 🧑‍💻 Contributing  
Pull requests are welcome! If you’d like to add new features (like speech-to-text 🎤 or improved memory), feel free to fork and contribute.  

---

## 📜 License  
MIT License.  
