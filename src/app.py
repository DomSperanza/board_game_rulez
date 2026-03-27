"""
Module: app
Purpose:
Main entry point for the Streamlit application to provide a chatbot interface.
"""
import os
import warnings

os.environ["CHROMA_TELEMETRY_IMPL"] = "None"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from generation.gemini_client import get_answer
from ingestion.registry import list_registered_games, sync_from_chroma_if_registry_empty
from retrieval.search import search_rulebook

sync_from_chroma_if_registry_empty()
_game_choices = list_registered_games()
if not _game_choices:
    _game_choices = ["(Add a PDF via Flask or ingest.py first)"]

st.set_page_config(page_title="Board Game Rulez", page_icon="🎲", layout="centered")

st.title("🎲 Board Game Rulebot")
st.write("Ask any rule question about your favorite board games!")

game_name = st.selectbox("Select a Game", _game_choices)
if game_name.startswith("(Add a PDF"):
    st.warning("No games in the library yet. Use the Flask app (`python src/flask_app.py`) or `python src/ingest.py` to add a rulebook.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a rule question (e.g. How do I build a city?)"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching rules and thinking..."):
        context_chunks = search_rulebook(prompt, game_name)
        answer = get_answer(prompt, context_chunks)

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
