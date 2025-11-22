import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import config, rag_endpoints


def create_app(config_path: str = None) -> FastAPI:
    # initialize singleton config (if any)
    config.Config.instance(config_path)
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
    return app


# # Optionally create a default app for importers/uvicorn module:app usage without a config file
# # (this won't parse CLI args; it uses None as the config path)
# app = create_app(None)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    # create app with provided config path and run
    app = create_app(args.config)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
