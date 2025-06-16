import os
from dotenv import load_dotenv
import streamlit as st
import cohere
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import docx

# Load environment variables
load_dotenv()

# Fetch Cohere API Key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Streamlit page config
st.set_page_config(page_title="📄 Document Chatbot", layout="centered")
st.title("📄 Document Chatbot with Cohere (Chatbot Mode)")

# Initialize Cohere client
if not COHERE_API_KEY:
    st.error("Missing COHERE_API_KEY in environment. Please add it to your .env file.")
    st.stop()
co_client = cohere.Client(COHERE_API_KEY)

# Caching helpers
@st.cache_data
def load_text_from_file(uploaded_file) -> str:
    text = ""
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    return text

@st.cache_data
def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_data
def embed_chunks(_co_client, chunks: list[str]) -> list[list[float]]:
    response = _co_client.embed(texts=chunks, model="embed-english-v2.0")
    return response.embeddings

# Retrieval & response

def get_top_chunks(co_client, query: str, chunks: list[str], embeddings: list[list[float]], k: int = 3) -> list[str]:
    q_emb = co_client.embed(texts=[query], model="embed-english-v2.0").embeddings[0]
    sims = cosine_similarity([q_emb], embeddings)[0]
    top_k = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k]


def generate_response(co_client, context: str, query: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = co_client.chat(
        model="command-nightly",
        message=prompt
    )
    return response.text

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # stores dicts with role and content

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

# File upload section
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
if uploaded_file:
    with st.spinner("Processing document..."):
        text = load_text_from_file(uploaded_file)
        st.session_state.chunks = split_text(text)
        st.session_state.embeddings = embed_chunks(co_client, st.session_state.chunks)
    st.success("Document processed and ready to chat!")

# Chat interface
if st.session_state.chunks:
    # Display chat history
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    # Input
    user_input = st.chat_input("Type your question...")
    if user_input:
        # Append user message
        st.session_state.history.append({"role": "user", "content": user_input})

        # Retrieve relevant chunks
        top_chunks = get_top_chunks(
            co_client,
            user_input,
            st.session_state.chunks,
            st.session_state.embeddings,
        )
        context = "\n---\n".join(top_chunks)

        # Generate and append assistant response
        with st.spinner("Thinking..."):
            answer = generate_response(co_client, context, user_input)
        st.session_state.history.append({"role": "assistant", "content": answer})

        # Rerun to display updated chat
        st.rerun()
else:
    st.info("Please upload a document to begin chatting.")