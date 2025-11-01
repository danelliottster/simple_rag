"""
llm.py

This module contains functions to build a context from chunks and send a prompt along with the context to an LLM using PydanticAI.
"""

from typing import List, Dict
from pydantic_ai import Agent

def organize_chunks(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Organize chunks by their source document and in order of their appearance.
    Args:
        chunks: A list of chunk dictionaries containing 'source_file' and 'chunk_index' and 'chunk_text'.
    Returns:
        A dictionary mapping source document names to lists of ordered chunk dictionaries.
    """
    organized = {}
    source_files = set([chunk.get('source_file') for chunk in chunks])
    for source_file in source_files:
        organized[source_file] = [chunk for chunk in chunks if chunk.get('source_file') == source_file]
        organized[source_file] = sorted(organized[source_file], key=lambda x: x['chunk_index'])
    return organized

def organized_context(chunks: List[Dict]) -> str:
    """
    Build a context string from the provided chunks, organizing them by source document.
    """
    organized_chunks = organize_chunks(chunks)
    context = ""
    for source_file, file_chunks in organized_chunks.items():
        context += f"BEGIN Source: {source_file}\n"
        context += "\n".join([chunk.get('chunk_text', '') for chunk in file_chunks])
        context += f"\nEND Source: {source_file}\n\n"
    return context.strip()

def build_context(chunks: List[Dict], max_tokens: int = 2048) -> str:
    """Build a context string from the provided chunks, ensuring it fits within the max token limit.

    Args:
        chunks: A list of chunk dictionaries containing 'chunk_text'.
        max_tokens: The maximum number of tokens the context can have.

    Returns:
        A string containing the concatenated chunk texts within the token limit.
    """
    context = ""
    token_count = 0

    ### grab as many chunks as fit within max_tokens
    chunks_to_keep = []
    for chunk in chunks:
        chunk_text = chunk.get('chunk_text', '')
        chunk_tokens = len(chunk_text.split())
        if token_count + chunk_tokens > max_tokens:
            break
        token_count += chunk_tokens
        chunks_to_keep.append(chunk)
    ### build a pretty context string from the kept chunks
    context = organized_context(chunks_to_keep)
    ### fin!
    return context

def send_to_llm(prompt: str, context: str, instructions: str, model: any) -> str:
    """Send a prompt and context to the LLM using PydanticAI and return the response.

    Args:
        prompt: The user prompt to send to the LLM.
        context: The context string to include in the LLM input.
        instructions: The instructions for the LLM.
        model: The LLM model to use provided by PydanticAI.

    Returns:
        The response from the LLM as a string.
    """
    full_prompt = f"Context:\n{context}\n\nPrompt:\n{prompt}"

    try:
        agent = Agent(model, instructions=instructions)
        result = agent.run_sync(full_prompt)
        return result.output
    except Exception as e:
        return f"Error communicating with LLM: {e}"