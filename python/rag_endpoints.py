"""
FastAPI API wrapping conversation methods from RagSqliteDB.

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
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Header, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import argparse
import os
import logging

# LLM and embedding imports
from google import genai
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import embed_chunks, config
from rag_sqlite_factory import get_rag_sqlite_db
import llm

router = APIRouter()

# Pydantic models
class NewConversationRequest(BaseModel):
    username: str

class NewConversationResponse(BaseModel):
    conversation_id: int

class ListConversationsResponse(BaseModel):
    conversations: List[Dict[str, Any]]

class GetConversationResponse(BaseModel):
    conversation_id: int
    conversation: List[Dict[str, Any]]

class DeleteConversationResponse(BaseModel):
    conversation_id: int
    deleted: bool

class UpdateConversationRequest(BaseModel):
    prompt: str

class UpdateConversationResponse(BaseModel):
    conversation_id: int
    conversation: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    error: str

def _get_rag_db():
    """Helper to obtain a RagSqliteDB instance from the singleton factory.

    Reads the runtime config to get the configured `db_path`. Raises HTTPException
    (500) when `db_path` is not configured.
    """
    cfg = config.get_config()
    db_path = cfg.get('db_path')
    if not db_path:
        raise HTTPException(status_code=500, detail="db_path not configured")
    return get_rag_sqlite_db(db_path)

def read_api_key(cfg):
    api_key_file = cfg.get('api_key_file')
    if not api_key_file or not os.path.exists(api_key_file):
        raise HTTPException(status_code=500, detail="api_key_file missing or not configured")
    try:
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read api_key_file: {e}")
    return api_key


def verify_api_token(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_token: Optional[str] = Header(None, alias="X-API-Token"),
    api_token: Optional[str] = Query(None, alias="api_token")
) -> str:
    """Dependency to verify API token from various sources.

    The expected token is read from config key 'api_token'. Incoming token is accepted
    from (in order): Authorization: Bearer <token>, X-API-Token header, query param
    'api_token'. If the configured token is missing,
    a 500 is returned. If the token is missing/invalid, a 401 is returned.
    """
    cfg = config.get_config()
    expected = cfg.get('api_token')
    if not expected:
        raise HTTPException(status_code=500, detail="api_token not configured")

    token = None
    # Authorization: Bearer <token>
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1].strip()

    # X-API-Token header or query param
    if not token:
        token = x_api_token or api_token

    if not token or token != expected:
        raise HTTPException(status_code=401, detail="invalid or missing api_token")

    return token

@router.post('/conversations/new', response_model=NewConversationResponse, status_code=201)
def new_conversation(
    request: NewConversationRequest,
    token: str = Depends(verify_api_token)
):
    db = _get_rag_db()
    conv_id = db.create_conversation(request.username)
    return NewConversationResponse(conversation_id=conv_id)


@router.get('/conversations', response_model=ListConversationsResponse)
def list_conversations(
    username: str = Query(..., description="Username to filter conversations"),
    token: str = Depends(verify_api_token)
):
    db = _get_rag_db()
    result = db.get_conversations_for_user(username)
    return ListConversationsResponse(conversations=result)


@router.get('/conversations/{conv_id}', response_model=GetConversationResponse)
def get_conversation(
    conv_id: int = Path(..., description="Conversation ID"),
    token: str = Depends(verify_api_token)
):
    db = _get_rag_db()
    conv = db.load_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return GetConversationResponse(conversation_id=conv_id, conversation=conv)


@router.post('/conversations/{conv_id}/delete', response_model=DeleteConversationResponse)
def delete_conversation(
    conv_id: int = Path(..., description="Conversation ID"),
    token: str = Depends(verify_api_token)
):
    """Soft-delete: clear conversation contents but keep the row (use update_conversation with empty list)."""
    db = _get_rag_db()
    # load to check exists
    conv = db.load_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    # soft-delete by setting deleted flag on the conversation row
    db.soft_delete_conversation(conv_id)
    return DeleteConversationResponse(conversation_id=conv_id, deleted=True)


@router.post('/conversations/{conv_id}/update', response_model=UpdateConversationResponse)
def update_conversation(
    request: UpdateConversationRequest,
    conv_id: int = Path(..., description="Conversation ID"),
    token: str = Depends(verify_api_token)
):
    """Handle a user prompt: embed the prompt, retrieve context, send to LLM,
    append the LLM response to the conversation, save it, and return the updated conversation.

    JSON body: {"prompt": "..."}
    """
    cfg = config.get_config()
    api_key = read_api_key(cfg)

    db = _get_rag_db()

    provider = GoogleProvider(api_key=api_key)
    model_light = GoogleModel(cfg.get('llm_model', 'gemini-2.5-flash'), provider=provider)

    ############################################################################
    # START load the conversation and add the latest user prompt

    # Load conversation (respecting soft-delete behavior)
    conv = db.load_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    # Append the user's prompt to conversation history
    conv.append({"role": "user", "content": request.prompt})

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
        raise HTTPException(status_code=404, detail="conversation record not found")
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
        raise HTTPException(status_code=500, detail=f"vector search failed: {e}")
    
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
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # Append LLM response and persist
    conv.append({"role": "llm", "content": response})
    try:
        db.update_conversation(conv_id, conv)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"failed to save conversation {conv_id}: {e}") 
        raise HTTPException(status_code=500, detail=f"failed to save conversation: {e}")
    
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

    return UpdateConversationResponse(conversation_id=conv_id, conversation=conv)
