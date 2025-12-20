# kondoo/rag-bot-template/app.py
import os
import logging
import yaml
from flask import Flask, request, jsonify
from flask_cors import CORS

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate
)
from llama_index.core.memory import ChatMemoryBuffer

# --- Provider-specific imports ---
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

cors_env = os.environ.get('CORS_ALLOWED_ORIGINS', '*')

if cors_env == '*':
    cors_origins = '*'
    logging.warning("⚠️ CORS configurado para permitir TODO (*). Revisa tu .env en producción.")
else:
    cors_origins = [origin.strip() for origin in cors_env.split(',')]
    logging.info(f"🛡️ CORS habilitado solo para: {cors_origins}")

# Aplicamos la configuración
CORS(app, origins=cors_origins)

CORS(app, origins=cors_origins)

# Global State
query_engine = None
index_storage = None # To hold the loaded index
session_store = {} # { 'session_id': ChatMemoryBuffer }


# --- SYSTEM PROMPT BUILDER ---
def build_system_prompt(persona_path, behavior_path):
    """
    Reads identity (YAML) and behavior (TXT) to merge them
    into a single master system prompt.
    """
    # 1. Load Identity (Persona) - Bot-specific data
    try:
        logging.info(f"Loading Persona from: {persona_path}")
        with open(persona_path, 'r', encoding='utf-8') as f:
            persona_data = yaml.safe_load(f) or {}
        
        # Convert YAML dictionary to a readable text block
        persona_str = "--- ASSISTANT IDENTITY ---\n"
        for key, value in persona_data.items():
            # Capitalize keys for better readability by the LLM (e.g., Name: Mavi)
            persona_str += f"{key.capitalize()}: {value}\n"
            
    except Exception as e:
        logging.warning(f"Could not load Persona file ({e}). Using default identity.")
        persona_str = "You are a helpful virtual assistant for SysAdminCtl.\n"

    # 2. Load Behavior - Global Rules
    try:
        logging.info(f"Loading Behavior from: {behavior_path}")
        with open(behavior_path, 'r', encoding='utf-8') as f:
            behavior_str = f.read()
    except Exception as e:
        logging.warning(f"Could not load Behavior file ({e}). Using default rules.")
        behavior_str = "Always respond professionally and base your answers on the context provided."

    # 3. Merge (Persona first to establish immediate context)
    final_system_prompt = (
        f"{persona_str}\n"
        f"--- BEHAVIORAL GUIDELINES ---\n"
        f"{behavior_str}\n"
    )
    
    return final_system_prompt

def initialize_query_engine():
    global query_engine, index_storage


    try:
        # --- 0. PRE-LOAD CONFIGURATION VARIABLES (With Defaults) ---
        # Definimos las variables aquí para poder loguearlas antes de iniciar
        llm_provider = os.environ.get('ANSWER_LLM_PROVIDER', 'gemini').lower()
        llm_model_name = os.environ.get('LLM_MODEL_NAME', 'Not Set')
        llm_base_url = os.environ.get('LLM_BASE_URL', 'N/A')
        
        # Enmascarar API Key por seguridad
        raw_api_key = os.environ.get('LLM_API_KEY')
        api_key_status = "**** (Set)" if raw_api_key else "None (Missing!)"
        if llm_provider == 'ollama_compatible': api_key_status = "Not Required (Ollama)"

        knowledge_provider = os.environ.get('KNOWLEDGE_PROVIDER', 'ollama').lower()
        embed_model_name = os.environ.get('EMBEDDING_MODEL_NAME', 'Not Set')
        embed_model_name = os.environ.get('EMBEDDING_MODEL_NAME', 'Not Set')
        ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')

        # --- New Configuration: Temperature & Top-K ---
        # Default Temperature: 0.1 (Focused)
        llm_temperature = float(os.environ.get('LLM_TEMPERATURE', '0.1'))
        # Default Top-K: 2 (Standard context window)
        rag_top_k = int(os.environ.get('RAG_TOP_K', '2'))

        knowledge_dir = os.environ.get('KNOWLEDGE_DIR', '/app/knowledge')
        persona_file = os.environ.get('BOT_PERSONA_FILE', '/app/config/persona.yaml')
        behavior_file = os.environ.get('BOT_BEHAVIOR_FILE', '/app/config/behavior.txt')

        # --- 1. PRINT STARTUP SUMMARY ---
        logging.info("\n" + "="*50)
        logging.info(" 🚀 KONDOO INITIALIZATION SUMMARY")
        logging.info("="*50)
        logging.info(f" 🧠  ANSWER ENGINE (LLM):")
        logging.info(f"     • Provider:      {llm_provider}")
        logging.info(f"     • Model Name:    {llm_model_name}")
        logging.info(f"     • API Key:       {api_key_status}")
        if llm_provider == 'ollama_compatible':
            logging.info(f"     • Base URL:      {llm_base_url}")
        
        logging.info(f"\n 📚  KNOWLEDGE ENGINE (RAG):")
        logging.info(f"     • Provider:      {knowledge_provider}")
        logging.info(f"     • Embed Model:   {embed_model_name}")
        logging.info(f"     • Vector Dir:    {knowledge_dir}")
        if knowledge_provider == 'ollama':
            logging.info(f"     • Ollama URL:    {ollama_base_url}")
        
        logging.info(f"\n ⚙️  FINE TUNING:")
        logging.info(f"     • Temperature:   {llm_temperature}")
        logging.info(f"     • Top-K:         {rag_top_k}")

        logging.info(f"\n 🎭  BOT IDENTITY:")
        logging.info(f"     • Persona File:  {persona_file}")
        logging.info(f"     • Behavior File: {behavior_file}")
        logging.info("="*50 + "\n")

        # --- 2. Configure LLM (The "Answer Engine") ---
        # Ahora usamos las variables que ya cargamos arriba
        if llm_provider == 'gemini':
            if not raw_api_key: raise ValueError("LLM_API_KEY needed for Gemini.")
            Settings.llm = GoogleGenAI(api_key=raw_api_key, model_name=llm_model_name, temperature=llm_temperature)
        elif llm_provider == 'openai':
            if not raw_api_key: raise ValueError("LLM_API_KEY needed for OpenAI.")
            Settings.llm = OpenAI(api_key=raw_api_key, model=llm_model_name, temperature=llm_temperature)
        elif llm_provider == 'ollama_compatible':
            if not llm_base_url or llm_base_url == 'N/A': raise ValueError("LLM_BASE_URL is required for ollama_compatible.")
            # Nota: pasamos 'ollama' como api_key dummy si no hay una real, para que el cliente no se queje
            Settings.llm = OpenAILike(model=llm_model_name, api_base=llm_base_url, api_key=raw_api_key or 'ollama', is_chat_model=True, temperature=llm_temperature)
        else:
            raise ValueError(f"Unsupported provider: {llm_provider}")

        # --- 3. Configure Embedding Model ---
        if knowledge_provider == 'ollama':
            Settings.embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=ollama_base_url)
        elif knowledge_provider == 'local':
            Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        elif knowledge_provider == 'openai':
            Settings.embed_model = OpenAIEmbedding(model=embed_model_name, api_key=raw_api_key)
        
        # --- 4. Load Knowledge Base ---
        if not os.path.exists(knowledge_dir) or not os.listdir(knowledge_dir):
             raise FileNotFoundError(f"Knowledge dir empty or not found: {knowledge_dir}")
        
        logging.info(f"Loading vectors...") # Mensaje simplificado porque ya lo dijimos arriba
        storage_context = StorageContext.from_defaults(persist_dir=knowledge_dir)
        index_storage = load_index_from_storage(storage_context)
        
        # --- 5. BUILD THE NEW PROMPT ---
        logging.info("Constructing Brain...")
        full_system_prompt = build_system_prompt(persona_file, behavior_file)
        
        qa_template_str = (
            f"{full_system_prompt}\n"
            "---------------------\n"
            "Context Information (Knowledge Base):\n"
            "{context_str}\n"
            "---------------------\n"
            "User Question: {query_str}\n"
            "Your Answer: "
        )
        qa_template = PromptTemplate(qa_template_str)

        # --- 6. Create Query Engine (Default for stateless requests) ---
        query_engine = index_storage.as_query_engine(
            streaming=False,
            text_qa_template=qa_template,
            similarity_top_k=rag_top_k
        )
        logging.info("✅ Kondoo Engine initialized successfully!")
        return True

    except Exception as e:
        logging.error(f"FATAL initialization error: {e}")
        return False

# --- Endpoints ---
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/query", methods=["POST"])
def process_query():
    if not query_engine:
        return jsonify({"error": "Engine not ready"}), 503

    data = request.get_json()
    user_query = data.get("query")
    session_id = data.get("session_id") # Optional: For chat history

    if not user_query:
        return jsonify({"error": "Missing query"}), 400

    logging.info(f"Query: '{user_query}' | Session: {session_id if session_id else 'None'}")

    try:
        # --- LOGIC SELECTION: CHAT (Stateful) vs QUERY (Stateless) ---
        if session_id:
            # 1. Get or Create Memory
            if session_id not in session_store:
                logging.info(f"Creating new memory buffer for session: {session_id}")
                session_store[session_id] = ChatMemoryBuffer.from_defaults(token_limit=3000)
            
            user_memory = session_store[session_id]

            # 2. Create Chat Engine for this request
            # We use 'context' mode to mix RAG with history
            # Re-fetch config to ensure consistency using the global helper or just reuse the logic?
            # Better approach: initialize_query_engine already set up the LLM in Settings.
            
            # Custom System Prompt for Chat
            # We need to re-read the files or store the prompt string globally. 
            # Optimization: Let's accept that we might re-read or we can store 'full_system_prompt' globally.
            # For now, let's rely on LlamaIndex defaults + context, but we want the Persona.
            
            # Let's grab the prompt from the current query_engine to avoid re-reading files? 
            # Or just re-build it since files are small. Let's call build_system_prompt again for safety/simplicity
            persona_file = os.environ.get('BOT_PERSONA_FILE', '/app/config/persona.yaml')
            behavior_file = os.environ.get('BOT_BEHAVIOR_FILE', '/app/config/behavior.txt')
            chat_system_prompt = build_system_prompt(persona_file, behavior_file)

            # Re-init engine with correct prompt
            # We use 'condense_plus_context' mode which is better for maintaining conversation history
            # It condenses the conversation history and the latest question into a standalone question
            chat_engine = index_storage.as_chat_engine(
                chat_mode="condense_plus_context",
                memory=user_memory,
                system_prompt=chat_system_prompt,
                similarity_top_k=int(os.environ.get('RAG_TOP_K', '2'))
            )
            
            response = chat_engine.chat(user_query)
        
        else:
            # Stateless (Legacy)
            response = query_engine.query(user_query)
        
        # Context Logging
        logging.info("--- Context Chunks ---")
        if response.source_nodes:
            for i, node in enumerate(response.source_nodes):
                cleaned_text = node.text.replace('\n', ' ')
                logging.info(f"Chunk {i+1} ({node.score:.2f}): '{cleaned_text[:200]}...'") 
        else:
            logging.info("No context chunks.")

        response_text = str(response)
        return jsonify({"response": response_text})

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if initialize_query_engine():
        app.run(host='0.0.0.0', port=5000)
    else:
        logging.critical("Failed to start.")