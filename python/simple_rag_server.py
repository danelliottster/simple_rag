import argparse, logging
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import config, rag_endpoints


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=5000)
args = parser.parse_args()

# init singleton config
config.Config.instance(args.config)
# basic logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Conversation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_endpoints.router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
