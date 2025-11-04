# kondoo/rag-bot-template/app.py
import os
import logging
from flask import Flask, request, jsonify

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate
)

# --- Provider-specific imports ---
# Import all the classes we might need based on the provider selection.
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike # For OpenAI-compatible APIs

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global variable for the query engine ---
query_engine = None

def initialize_query_engine():
    """
    Initializes the RAG query engine based on provider settings
    from environment variables.
    """
    global query_engine

    try:
        # --- 1. Load Provider Selection ---
        llm_provider = os.environ.get('ANSWER_LLM_PROVIDER', 'gemini').lower()
        embedding_provider = os.environ.get('KNOWLEDGE_PROVIDER', 'ollama').lower()
        logging.info(f"Using Answer Engine Provider: '{llm_provider}'")
        logging.info(f"Using Knowledge Embedding Provider: '{embedding_provider}'")

        # --- 2. Configure LLM (The "Answer Engine") ---
        # (Esta sección no cambia)
        llm_model_name = os.environ.get('LLM_MODEL_NAME')
        llm_api_key = os.environ.get('LLM_API_KEY')

        if llm_provider == 'gemini':
            if not llm_api_key: raise ValueError("LLM_API_KEY is required for the Gemini provider.")
            Settings.llm = Gemini(api_key=llm_api_key, model_name=llm_model_name)
        elif llm_provider == 'openai':
            if not llm_api_key: raise ValueError("LLM_API_KEY is required for the OpenAI provider.")
            Settings.llm = OpenAI(api_key=llm_api_key, model=llm_model_name)
        elif llm_provider == 'ollama_compatible':
            llm_base_url = os.environ.get('LLM_BASE_URL')
            if not llm_base_url: raise ValueError("LLM_BASE_URL is required for the ollama_compatible provider.")
            Settings.llm = OpenAILike(model=llm_model_name, api_base=llm_base_url, api_key=llm_api_key, is_chat_model=True)
        else:
            raise ValueError(f"Unsupported ANSWER_LLM_PROVIDER: '{llm_provider}'")
        logging.info(f"Answer Engine '{llm_model_name}' configured successfully.")

        # --- 3. Configure Embedding Model (The "Knowledge Source") ---
        # (Esta sección no cambia)
        embed_model_name = os.environ.get('EMBEDDING_MODEL_NAME')
        if embedding_provider == 'ollama':
            ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')
            Settings.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=ollama_base_url)
        elif embedding_provider == 'local':
            Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        elif embedding_provider == 'openai':
            if not llm_api_key: raise ValueError("LLM_API_KEY is required for the OpenAI embedding provider.")
            Settings.embed_model = OpenAIEmbedding(model=embed_model_name, api_key=llm_api_key)
        else:
            raise ValueError(f"Unsupported KNOWLEDGE_PROVIDER: '{embedding_provider}'")
        logging.info(f"Knowledge Embedding model '{embed_model_name}' configured successfully.")
        
        # --- 4. Load the Knowledge Base ---
        # (Esta sección no cambia)
        knowledge_dir = os.environ.get('KNOWLEDGE_DIR', '/app/knowledge')
        if not os.path.exists(knowledge_dir) or not os.listdir(knowledge_dir):
             raise FileNotFoundError(f"Knowledge base directory is empty or not found at '{knowledge_dir}'")
        logging.info(f"Loading knowledge base from '{knowledge_dir}'...")
        storage_context = StorageContext.from_defaults(persist_dir=knowledge_dir)
        index = load_index_from_storage(storage_context)
        
        # --- 5. Load Personality and Create Final Prompt Template ---
        logging.info("Loading personality and building final prompt...")
        personality_path = os.environ.get('BOT_PERSONALITY_FILE', '/app/personality.txt')
        try:
            with open(personality_path, 'r', encoding='utf-8') as f:
                bot_personality = f.read()
            logging.info(f"Successfully loaded personality from {personality_path}")
        except Exception as e:
            logging.error(f"Error reading personality file: {e}. Using default personality.")
            bot_personality = "Eres un asistente servicial."

        # Esta es la "plantilla RAG" que une todo.
        qa_template_str = (
            f"{bot_personality}\n"
            "---------------------\n"
            "Contexto: {context_str}\n"
            "---------------------\n"
            "Pregunta: {query_str}\n"
            "Respuesta: "
        )
        qa_template = PromptTemplate(qa_template_str)

        # --- 6. Create the Query Engine (with the new template) ---
        logging.info("Creating query engine with custom personality template...")
        query_engine = index.as_query_engine(
            streaming=False,
            text_qa_template=qa_template  # <-- Aquí aplicamos nuestra plantilla personalizada
        )
        logging.info("✅ RAG Query Engine initialized successfully!")
        return True

    except Exception as e:
        logging.error(f"FATAL: An error occurred during engine initialization: {e}")
        return False

# The Flask endpoints (/health, /query) and the main execution block
# remain exactly the same, as they are provider-agnostic.

@app.route("/health", methods=["GET"])
def health_check():
    # ... (code unchanged)
    return jsonify({"status": "ok"}), 200

@app.route("/query", methods=["POST"])
def process_query():
    """
    Processes a user query against the RAG engine.
    """
    if not query_engine:
        logging.error("Query received but engine is not initialized.")
        return jsonify({"error": "Query engine is not available"}), 503

    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    logging.info(f"Received query: '{user_query}'")

    try:
        response = query_engine.query(user_query)
        response_text = str(response)
        logging.info(f"Generated response: '{response_text}'")
        return jsonify({"response": response_text})

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return jsonify({"error": "Failed to process the query"}), 500

if __name__ == '__main__':
    if initialize_query_engine():
        app.run(host='0.0.0.0', port=5000)
    else:
        logging.critical("Application startup failed due to initialization error.")