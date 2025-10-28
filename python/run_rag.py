import rag_sqlite

rag_db = rag_sqlite.RagSqliteDB(db_path="/home/dane/hugo/hugo.db")
rag_db.load_chunks()