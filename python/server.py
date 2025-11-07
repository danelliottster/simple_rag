"""
Simple Flask API wrapping conversation methods from RagSqliteDB.

Endpoints:
 - POST /conversations/new             {"username": "alice"} -> {"conversation_id": 1}
 - GET  /conversations/<id>           -> {"conversation": [...]}
 - GET  /conversations?username=...   -> list of conversations for user
 - POST /conversations/<id>/delete    soft-delete (clears conversation)
 - POST /conversations/<id>/append    {"system_output": "..."} -> appends an LLM message
 - POST /conversations/save           {"conversation_id": optional, "username": optional, "conversation": [...]} -> saves conversation

This module uses existing methods on RagSqliteDB: create_conversation, load_conversation,
update_conversation, get_conversations_for_user.
"""
from flask import Flask, request, jsonify
import argparse
import os
import logging

from rag_sqlite import RagSqliteDB
import config

# LLM and embedding imports
from google import genai
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import embed_chunks, rag_sqlite
import llm

logger = logging.getLogger(__name__)

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=5000)
args = parser.parse_args()
# init singleton config
config.Config.instance(args.config)
# basic logging
logging.basicConfig(level=logging.INFO)
config.Config.instance(args.config)
cfg = config.get_config()

rag_db = rag_sqlite.RagSqliteDB(db_path=cfg.get('db_path'))
# get the path to the db directory
db_dir = os.path.dirname(rag_db.db_path)
# load the "vector DB"
rag_db.load_index_file(db_dir+"/model.pkl")

def get_db(cfg):
    db_path = cfg.get('db_path')
    if not db_path:
        raise RuntimeError('db_path not configured')
    return RagSqliteDB(db_path=db_path)


@app.route('/conversations/new', methods=['POST'])
def new_conversation():
    data = request.get_json(force=True) or {}
    username = data.get('username')
    if not username:
        return jsonify({"error": "username required"}), 400
    cfg = config.get_config()
    db = get_db(cfg)
    conv_id = db.create_conversation(username)
    return jsonify({"conversation_id": conv_id}), 201


@app.route('/conversations', methods=['GET'])
def list_conversations():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "username query parameter required"}), 400
    cfg = config.get_config()
    db = get_db(cfg)
    result = db.get_conversations_for_user(username)
    return jsonify({"conversations": result})


@app.route('/conversations/<int:conv_id>', methods=['GET'])
def get_conversation(conv_id: int):
    cfg = config.get_config()
    db = get_db(cfg)
    conv = db.load_conversation(conv_id)
    if conv is None:
        return jsonify({"error": "conversation not found"}), 404
    return jsonify({"conversation_id": conv_id, "conversation": conv})


@app.route('/conversations/<int:conv_id>/delete', methods=['POST'])
def delete_conversation(conv_id: int):
    """Soft-delete: clear conversation contents but keep the row (use update_conversation with empty list)."""
    cfg = config.get_config()
    db = get_db(cfg)
    # load to check exists
    conv = db.load_conversation(conv_id)
    if conv is None:
        return jsonify({"error": "conversation not found"}), 404
    # soft-delete by setting deleted flag on the conversation row
    db.soft_delete_conversation(conv_id)
    return jsonify({"conversation_id": conv_id, "deleted": True})


@app.route('/conversations/<int:conv_id>/update', methods=['POST'])
def update_conversation(conv_id: int):
    """Handle a user prompt: embed the prompt, retrieve context, send to LLM,
    append the LLM response to the conversation, save it, and return the updated conversation.

    JSON body: {"prompt": "..."}
    """
    data = request.get_json(force=True) or {}
    prompt = data.get('prompt')
    if prompt is None:
        return jsonify({"error": "prompt required"}), 400

    cfg = config.get_config()
    db = get_db(cfg)

    # Load conversation (respecting soft-delete behavior)
    conv = db.load_conversation(conv_id)
    if conv is None:
        return jsonify({"error": "conversation not found"}), 404
    
    conv_record = db.get_conversation_record(conv_id)
    if conv_record is None:
        return jsonify({"error": "conversation record not found"}), 404
    summary = conv_record.get('summary')

    # Append the user's prompt to conversation history
    conv.append({"role": "user", "content": prompt})

    # Build user-only prompt history for embedding (same as run_rag)
    user_prompt_history = ""
    for msg in conv:
        if msg.get('role') == 'user':
            user_prompt_history += f"{msg.get('content')}\n"

    # Initialize LLM and embedding client from config
    api_key_file = cfg.get('api_key_file')
    if not api_key_file or not os.path.exists(api_key_file):
        return jsonify({"error": "api_key_file missing or not configured"}), 500
    try:
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
    except Exception as e:
        return jsonify({"error": f"failed to read api_key_file: {e}"}), 500

    client = genai.Client(api_key=api_key)
    provider = GoogleProvider(api_key=api_key)
    model_light = GoogleModel(cfg.get('llm_model', 'gemini-2.5-flash'), provider=provider)

    # Embed the prompt and search the vector DB for relevant chunks
    try:
        prompt_embedding = embed_chunks.batch_embed_chunks(
            [{'chunk': user_prompt_history}],
            client=client
        )[0]['embedding']
    except Exception as e:
        return jsonify({"error": f"embedding failed: {e}"}), 500

    tags = cfg.get('tags', []) or []
    top_k = cfg.get('max_context_documents', 20)
    try:
        relevant_chunks = db.search_chunks(embedding=prompt_embedding, top_k=top_k, tags=tags)
    except Exception as e:
        return jsonify({"error": f"vector search failed: {e}"}), 500

    # Build a context string for the LLM
    try:
        context = llm.build_context(chunks=relevant_chunks or [], max_tokens=cfg.get('max_context_tokens', 20000))
    except Exception as e:
        context = ""

    # Build the full prompt (include both user and llm messages)
    full_prompt = ""
    for msg in conv:
        if msg.get('role') == 'user':
            full_prompt += f"User: {msg.get('content')}\n"
        else:
            full_prompt += f"LLM: {msg.get('content')}\n"

    # Send to LLM
    try:
        instructions = cfg.get('instructions', '')
        response = llm.send_to_llm(prompt=full_prompt, context=context, instructions=instructions, model=model_light)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 500

    # Append LLM response and persist
    conv.append({"role": "llm", "content": response})
    try:
        db.update_conversation(conv_id, conv)
    except Exception as e:
        logger.error(f"failed to save conversation {conv_id}: {e}") 
        return jsonify({"error": f"failed to save conversation: {e}"}), 500
    
    # update the summary
    if not summary:
        # extract the first user prompt from the conversation and assign to summary
        summary = next((m.get('content') for m in conv if m.get('role') == 'user'), None)
        if summary and len(summary) > 50:
            summary = summary[:50]
        db.update_summary(conv_id, summary)

    return jsonify({"conversation_id": conv_id, "conversation": conv})


# @app.route('/conversations/save', methods=['POST'])
# def save_conversation():
#     """Save a conversation to the DB. Accepts either an existing conversation_id to update
#     or a username to create a new conversation and save content.

#     JSON body options:
#       - {"conversation_id": 1, "conversation": [...]}  -> updates existing
#       - {"username": "alice", "conversation": [...]} -> creates and saves
#     Returns: {"conversation_id": id}
#     """
#     data = request.get_json(force=True) or {}
#     conv = data.get('conversation')
#     if conv is None:
#         return jsonify({"error": "conversation (list) required in body"}), 400

#     cfg = config.get_config()
#     db = get_db(cfg)

#     conv_id = data.get('conversation_id')
#     if conv_id:
#         # ensure exists
#         _existing = db.load_conversation(conv_id)
#         if _existing is None:
#             return jsonify({"error": "conversation_id not found"}), 404
#         db.update_conversation(conv_id, conv)
#         return jsonify({"conversation_id": conv_id})

#     username = data.get('username')
#     if not username:
#         return jsonify({"error": "either conversation_id or username required"}), 400
#     # create and update
#     new_id = db.create_conversation(username)
#     db.update_conversation(new_id, conv)
#     return jsonify({"conversation_id": new_id}), 201

if __name__ == '__main__':
    app.run(host=args.host, port=args.port)
