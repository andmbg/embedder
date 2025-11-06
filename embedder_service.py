"""
Custom WhisperX service with support for WAV2VEC2_ASR_LARGE_LV60K_960H alignment model
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from dotenv import load_dotenv, find_dotenv

from src.sentence_transformer import EmbeddingService
from src.logger import logger

# Load environment variables from .env file
load_dotenv(find_dotenv())

logger.info(f"HF_TOKEN: {os.getenv('HF_TOKEN')}")

app = Flask(__name__)


@app.route("/embed", methods=["POST"])
def embeddings():
    """Get embeddings for a list of texts."""
    # --- API Key check ---
    api_key = request.headers.get("Authorization")
    expected_key = f"Bearer {os.getenv('API_TOKEN')}"
    if expected_key and api_key != expected_key:
        return (
            jsonify({"error": f"expected_key: {expected_key}; api_key: {api_key}"}),
            401,
        )
    # ----------------------

    embedder = None

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No transcript JSON provided"}), 400

        chunks = data.get("chunks")
        model_name = data.get("model", "all-MiniLM-L6-v2")

        if not chunks:
            return jsonify({"error": "No transcript provided"}), 400

        logger.info("Sending transcript to embedder")
        logger.debug("model_name: %s", model_name)

        embedder = EmbeddingService(model=model_name)
        result = embedder.embed_transcript(chunks=chunks)
        logger.debug("Embedding completed")

        # Thorough cleanup, as we had some OOM issues before:
        del chunks

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        import gc

        gc.collect()

        return jsonify(result)

    except Exception as e:
        logger.error(f"Embedding request processing error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if embedder:
            if hasattr(embedder, "cleanup"):
                embedder.cleanup()
            del embedder

        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 19000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Embedder service on {host}:{port}")
    app.run(host=host, port=port, debug=True, threaded=True)
