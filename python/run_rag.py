import argparse, os
import rag_sqlite

parser = argparse.ArgumentParser(description="Build RAG chunks from a corpus directory.")
parser.add_argument('--api_key_file', type=str, required=True, help='Path to text file containing Google API key')
parser.add_argument('--db_path', type=str, required=True, help='Path to sqlite database to store chunks and metadata')
parser.add_argument('--tags', nargs='*', default=[], help='Tags to filter chunks and documents')
args = parser.parse_args()

rag_db = rag_sqlite.RagSqliteDB(db_path=args.db_path)
# get the path to the db directory
db_dir = os.path.dirname(args.db_path)
# load the "vector DB"
rag_db.load_index_file(db_dir)

#######################################################
# START interactive loop for user to interact with RAG
while True:
    user_input = input("Enter your query (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    #######################################################
    # START embed prompt

    # END embed prompt
    ########################################################


    #########################################################
    # START search vector DB for relevant chunks

    # END search vector DB for relevant chunks
    #########################################################

    #########################################################
    # START build context from retrieved chunks

    # END build context from retrieved chunks
    #########################################################

    #########################################################
    # START send prompt + context to LLM and get response

    # END send prompt + context to LLM and get response
    #########################################################


    # TODO: Add RAG logic here to process user_input
    print("You entered:", user_input)
# END interactive loop for user to interact with RAG
#######################################################
