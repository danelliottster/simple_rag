import argparse, os
import rag_sqlite, embed_chunks, llm
from google import genai
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

parser = argparse.ArgumentParser(description="Build RAG chunks from a corpus directory.")
parser.add_argument('--api_key_file', type=str, required=True, help='Path to text file containing Google API key')
parser.add_argument('--db_path', type=str, required=True, help='Path to sqlite database to store chunks and metadata')
parser.add_argument('--tags', nargs='*', default=[], help='Tags to filter chunks and documents')
args = parser.parse_args()

rag_db = rag_sqlite.RagSqliteDB(db_path=args.db_path)
# get the path to the db directory
db_dir = os.path.dirname(args.db_path)
# load the "vector DB"
rag_db.load_index_file(db_dir+"/model.pkl")

###############################################
# START put together LLM interfaces
with open(args.api_key_file, 'r') as f:
    api_key = f.read().strip()
client = genai.Client(api_key=api_key)

provider = GoogleProvider(api_key=api_key)
model_light = GoogleModel("gemini-2.5-flash", provider=provider)
model_heavy = GoogleModel("gemini-2.5-pro", provider=provider)
# END put together LLM interfaces
################################################

#######################################################
# START interactive loop for user to interact with RAG
# Maintain conversation history for multi-turn support
conversation_history = []
while True:
    user_input = input("Enter your query (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    #######################################################
    # START build conversation history for multi-turn
    # Each turn: append user and LLM messages
    conversation_history.append({"role": "user", "content": user_input})
    # Build user-only prompt from history
    user_prompt_history = ""
    for msg in conversation_history:
        if msg["role"] == "user":
            user_prompt_history += f"{msg['content']}\n"
    # END build conversation history for multi-turn
    #########################################################

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
        tags=args.tags
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
    conversation_history.append({"role": "user", "content": user_input})
    # Build full prompt from history (user and LLM) for LLM input
    full_prompt = ""
    for msg in conversation_history:
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

    # Add LLM response to history
    conversation_history.append({"role": "llm", "content": response})

    print("You entered:", user_input)
    print("LLM response:", response)
# END interactive loop for user to interact with RAG
#######################################################
