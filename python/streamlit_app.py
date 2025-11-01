"""
Streamlit app for RAG-based QA using logic from run_rag.py.
Loads the database and vector index once, then allows interactive queries.
"""
import streamlit as st
import os
import rag_sqlite, embed_chunks, llm
from google import genai
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import argparse
from simple_rag.python.config import get_config

# instantiate config singleton and use it as defaults when CLI args are not provided
cfg = get_config()
api_key_file = cfg.get('api_key_file')
db_path = cfg.get('db_path')
tags = cfg.get('tags', []) or []

# --- Load DB and LLMs once ---
@st.cache_resource(show_spinner=True)
def load_resources(api_key_file, db_path):
    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()
    rag_db = rag_sqlite.RagSqliteDB(db_path=db_path)
    db_dir = os.path.dirname(db_path)
    rag_db.load_index_file(os.path.join(db_dir, "model.pkl"))
    client = genai.Client(api_key=api_key)
    provider = GoogleProvider(api_key=api_key)
    model_light = GoogleModel("gemini-2.5-flash", provider=provider)
    model_heavy = GoogleModel("gemini-2.5-pro", provider=provider)
    return rag_db, client, model_light, model_heavy

if api_key_file and db_path:
    rag_db, client, model_light, model_heavy = load_resources(api_key_file, db_path)
else:
    st.warning("Please provide API key file and DB path.")
    st.stop()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

st.title("RAG-based QA Streamlit App")

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    user_prompt_history = ""
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            user_prompt_history += f"{msg['content']}\n"

    #######################################################
    # START embed prompt
    # Use only user prompts for embedding
    prompt_embedding = embed_chunks.batch_embed_chunks(
        [{'chunk': user_prompt_history}],
        client=client
    )[0]['embedding']
    # END embed prompt
    ########################################################

    #########################################################
    # START search vector DB for relevant chunks
    relevant_chunks = rag_db.search_chunks(
        embedding=prompt_embedding,
        top_k=20,
        tags=tags
    )
    # END search vector DB for relevant chunks
    #########################################################

    #########################################################
    # START build context from retrieved chunks
    context = "\n\n".join([chunk['chunk_text'] for chunk in relevant_chunks])
    # END build context from retrieved chunks
    #########################################################

    #########################################################
    # START build conversation history for multi-turn
    # Each turn: append user and LLM messages
    # Build full prompt from history (user and LLM) for LLM input
    full_prompt = ""
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            full_prompt += f"User: {msg['content']}\n"
        else:
            full_prompt += f"LLM: {msg['content']}\n"
    # END build conversation history for multi-turn
    #########################################################

    #########################################################
    # START send prompt + context to LLM and get response
    response = llm.send_to_llm(
        prompt=full_prompt,
        context=context,
        model=model_light
    )
    # END send prompt + context to LLM and get response
    #########################################################

    #################################################
    # START Add LLM response to history
    st.session_state.conversation_history.append({"role": "llm", "content": response})
    # END Add LLM response to history

    #################################################
    # START Display in Streamlit chat
    # Display user message in chat message container
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
    # with st.chat_message("user"):
    #     st.markdown(prompt)
    # # Display assistant response in chat message container
    # with st.chat_message("assistant"):
    #     st.markdown(response)
    # END Display in Streamlit chat
    #################################################