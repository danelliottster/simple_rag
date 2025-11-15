"""
Flask API wrapping conversation methods from RagSqliteDB.

Endpoints implemented in this module:
 - POST   /conversations/new
            Body: {"username": "<username>"}
            Response: {"conversation_id": <id>} (201)
            Creates a new conversation for the given user.

 - GET    /conversations
            Query: ?username=<username>
            Response: {"conversations": [...]} (200)
            Lists conversations for the given user.

 - GET    /conversations/<id>
            Response: {"conversation_id": <id>, "conversation": [...]} (200)
            Loads a single conversation by id.

 - POST    /conversations/<id>/delete
            Response: {"conversation_id": <id>, "deleted": True} (200)
            Soft-deletes a conversation (sets deleted flag / clears contents depending on DB impl).

 - POST   /conversations/<id>/update
            Body: {"prompt": "<user prompt>"}
            Response: {"conversation_id": <id>, "conversation": [...]} (200)
            Embeds the prompt, retrieves relevant context, calls the LLM, appends the LLM response
            to the conversation, persists changes, and may update a brief conversation summary.

Notes:
 - This module relies on RagSqliteDB methods such as create_conversation, load_conversation,
   update_conversation, get_conversations_for_user, soft_delete_conversation, get_conversation_record,
   search_chunks, and update_summary.
 - LLM/embedding clients and model selection are configured via the project's config and
   the stored api_key file.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
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
CORS(app)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
parser.add_argument('--host', type=str, default='0.0.0.0')
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

def read_api_key(cfg):
    api_key_file = cfg.get('api_key_file')
    if not api_key_file or not os.path.exists(api_key_file):
        return jsonify({"error": "api_key_file missing or not configured"}), 500
    try:
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
    except Exception as e:
        return jsonify({"error": f"failed to read api_key_file: {e}"}), 500
    return api_key


def require_api_token(func):
    """Decorator to require api_token configured in config for each endpoint.

    The expected token is read from config key 'api_token'. Incoming token is accepted
    from (in order): Authorization: Bearer <token>, X-API-Token header, query param
    'api_token', or JSON body field 'api_token'. If the configured token is missing,
    a 500 is returned. If the token is missing/invalid, a 401 is returned.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        cfg = config.get_config()
        expected = cfg.get('api_token')
        if not expected:
            return jsonify({"error": "api_token not configured"}), 500

        token = None
        # Authorization: Bearer <token>
        auth = request.headers.get('Authorization')
        if auth and auth.startswith('Bearer '):
            token = auth.split(' ', 1)[1].strip()

        # X-API-Token header or query param
        if not token:
            token = request.headers.get('X-API-Token') or request.args.get('api_token')

        # JSON body
        if not token:
            try:
                body = request.get_json(silent=True) or {}
                token = body.get('api_token')
            except Exception:
                token = None

        if token != expected:
            return jsonify({"error": "invalid or missing api_token"}), 401

        return func(*args, **kwargs)

    return wrapper

@app.route('/conversations/new', methods=['POST'])
@require_api_token
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
@require_api_token
def list_conversations():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "username query parameter required"}), 400
    cfg = config.get_config()
    db = get_db(cfg)
    result = db.get_conversations_for_user(username)
    return jsonify({"conversations": result})


@app.route('/conversations/<int:conv_id>', methods=['GET'])
@require_api_token
def get_conversation(conv_id: int):
    cfg = config.get_config()
    db = get_db(cfg)
    conv = db.load_conversation(conv_id)
    if conv is None:
        return jsonify({"error": "conversation not found"}), 404
    return jsonify({"conversation_id": conv_id, "conversation": conv})


@app.route('/conversations/<int:conv_id>/delete', methods=['POST'])
@require_api_token
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
@require_api_token
def update_conversation(conv_id: int):
    """Handle a user prompt: embed the prompt, retrieve context, send to LLM,
    append the LLM response to the conversation, save it, and return the updated conversation.

    JSON body: {"prompt": "..."}
    """
    cfg = config.get_config()
    api_key = read_api_key(cfg)

    data = request.get_json(force=True) or {}
    prompt = data.get('prompt')
    if prompt is None:
        return jsonify({"error": "prompt required"}), 400

    cfg = config.get_config()
    db = get_db(cfg)
    provider = GoogleProvider(api_key=api_key)
    model_light = GoogleModel(cfg.get('llm_model', 'gemini-2.5-flash'), provider=provider)
    api_key = read_api_key(cfg)

    ############################################################################
    # START load the conversation and add the latest user prompt

    # Load conversation (respecting soft-delete behavior)
    conv = db.load_conversation(conv_id)
    if conv is None:
        return jsonify({"error": "conversation not found"}), 404

    # Append the user's prompt to conversation history
    conv.append({"role": "user", "content": prompt})

    # END load the conversation and add the latest user prompt
    ############################################################################

    ############################################################################
    # START embed conversation

    client = genai.Client(api_key=api_key)
    prompt_embedding = embed_chunks.embed_conversation(conv, client, user_only=True)

    # END embed converstation
    ############################################################################

    ############################################################################
    # START grab the existing conversation summary
    
    conv_record = db.get_conversation_record(conv_id)
    if conv_record is None:
        return jsonify({"error": "conversation record not found"}), 404
    summary = conv_record.get('conversation_summary')
        
    # END grab the existing conversation summary
    ############################################################################

    ############################################################################
    # START vector search for relevant chunks
    
    tags = cfg.get('tags', []) or []
    top_k = cfg.get('max_context_documents', 20)
    try:
        relevant_chunks = db.search_chunks(embedding=prompt_embedding, top_k=top_k, tags=tags)
    except Exception as e:
        return jsonify({"error": f"vector search failed: {e}"}), 500
    
    # END vector search for relevant chunks
    ############################################################################

    ############################################################################
    # START build context string for LLM
    
    try:
        context = llm.build_context(chunks=relevant_chunks or [], max_tokens=cfg.get('max_context_tokens', 20000))
    except Exception as e:
        context = ""    
    
    # END build context string for LLM
    ############################################################################

    ############################################################################
    # START build the full prompt
        
    full_prompt = ""
    for msg in conv:
        if msg.get('role') == 'user':
            full_prompt += f"User: {msg.get('content')}\n"
        else:
            full_prompt += f"LLM: {msg.get('content')}\n"

    # END build the full prompt
    ############################################################################

    ############################################################################
    # START send to LLM
    
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
    
    # END send to LLM
    ############################################################################
    
    ############################################################################
    # START optionally update the conversation summary
    
    if not summary:
        # extract the first user prompt from the conversation and assign to summary
        summary = next((m.get('content') for m in conv if m.get('role') == 'user'), None)
        if summary and len(summary) > 50:
            summary = summary[:50]
        db.update_summary(conv_id, summary)
        
    # END optionally update the conversation summary
    ############################################################################

    return jsonify({"conversation_id": conv_id, "conversation": conv})


if __name__ == '__main__':
    app.run(host=args.host, port=args.port)
