"""
llm.py

This module contains functions to build a context from chunks and send a prompt along with the context to an LLM using PydanticAI.
"""

from typing import List, Dict
from pydantic_ai import Agent

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

    for chunk in chunks:
        chunk_text = chunk.get('chunk_text', '')
        chunk_tokens = len(chunk_text.split())

        if token_count + chunk_tokens > max_tokens:
            break

        context += chunk_text + "\n"
        token_count += chunk_tokens

    return context.strip()

def send_to_llm(prompt: str, context: str, model: str = "openai:gpt-4") -> str:
    """Send a prompt and context to the LLM using PydanticAI and return the response.

    Args:
        prompt: The user prompt to send to the LLM.
        context: The context string to include in the LLM input.
        model: The LLM model to use (default is "openai:gpt-4").

    Returns:
        The response from the LLM as a string.
    """
    full_prompt = f"Context:\n{context}\n\nPrompt:\n{prompt}"

    try:
        instructions = (
            "You are a retrieval-augmented generation (RAG) system. Use the provided context to answer the user's query. "
            "If the context does not contain the information needed to answer the query, respond with 'I don't know based on the provided context.' "
            "Do not make up information or provide answers outside the given context."
        )

        agent = Agent(model, instructions=instructions)
        result = agent.run_sync(full_prompt)
        return result.output
    except Exception as e:
        return f"Error communicating with LLM: {e}"