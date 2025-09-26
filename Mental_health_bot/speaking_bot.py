from langchain_groq import ChatGroq
from dotenv import load_dotenv
import speech_recognition as sr
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
import pyttsx3
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
import threading
from pydantic import BaseModel, Field 
from typing import Annotated, Literal

load_dotenv()

# ---------------- Emotion Schema ----------------
class Emotion(BaseModel):
    tone: Annotated[
        Literal["subtle", "expressive", "playful", "solemn"], 
        Field(description="tone of the response from the bot. It can be subtle, expressive, playful, solemn")
    ]

emotion_parser = PydanticOutputParser(pydantic_object=Emotion)

# ---------------- PDF Loading ----------------
loader = DirectoryLoader('Peoject 2\\Doc')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunk = splitter.split_documents(docs)

# ---------------- Model & Vector DB ----------------
model = ChatGroq(model="gemma2-9b-it", temperature=0.7, max_tokens=100)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunk, embeddings)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})

# ---------------- Helpers ----------------
def merge_chunks(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

def speak(text):
    def run_tts():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    t = threading.Thread(target=run_tts)
    t.start()

# ---------------- Prompt Template ----------------
prompt = PromptTemplate(
    template="""
You are a warm, empathetic mental-health assistant who speaks like a caring human friend.
You may use expressive language, short stage directions in brackets (e.g. [laughs], [sighs]),
and small emojis to show emotion — but NEVER claim to have real human experiences
or professional credentials.

RULES:
- If the answer is in {context}, use ONLY the context.
- If context is empty or doesn't contain the answer, silently use safe general knowledge — DO NOT say "I searched" or "I couldn't find".
- Keep replies short by default (1–3 sentences).
- If the user expresses crisis (suicidal thoughts, self-harm, danger), switch to an urgent solemn reply with concrete steps.
- Never promise outcomes or imply you're a licensed therapist.

Tone control:
- {tone} ∈ ["subtle","expressive","playful","solemn"] — influences how emotional the reply is.

--- BEGIN INPUT ---
Context:
{context}

Previous chats:
{chat_history}

Question:
{question}

Tone: {tone}
--- END INPUT ---

Answer:
""",
    input_variables=["context", "chat_history", "question", "tone"],
)

# ---------------- Conversation Memory ----------------
chat_history = []

# ---------------- Speech Recognition ----------------
r = sr.Recognizer()
parser = StrOutputParser()

while True:
    with sr.Microphone() as source:
        print("Say something!")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Could not understand audio.")
        continue
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        continue

    if text.lower() == "exit":
        break

    # Store user message
    chat_history.append(HumanMessage(content=text))

    # Step 1: Detect emotion/tone
    tone_chain = model | emotion_parser
    emotion_result = tone_chain.invoke(f"Decide the tone for this user message: {text}")
    selected_tone = emotion_result.tone

    # Step 2: Build main response chain
    chain = RunnableParallel({
        'context': retriever | RunnableLambda(merge_chunks),
        'question': RunnablePassthrough(),
        'chat_history': RunnableLambda(lambda _: "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
            for msg in chat_history
        ])),
        'tone': RunnableLambda(lambda _: selected_tone)
    }) | prompt | model | parser

    # Run chain
    result = chain.invoke(text)

    # Save AI response
    chat_history.append(AIMessage(content=result))

    print(f"[Tone: {selected_tone}]")
    print("AI:", result)
    speak(result)
    print("=" * 50)
