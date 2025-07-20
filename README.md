# Build Your Own AI Clone

This project is a RAG-based Tech Advisor Chatbot using Pinecone, Groq, and Langchain.

## Installation & Setup

1. **Clone the repository and navigate to the project directory.**

2. **Create and activate a virtual environment:**
   - On Windows (PowerShell):
     ```sh
     python -m venv venv
     .\venv\Scripts\Activate
     ```
   - On macOS/Linux:
     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set your API keys as environment variables (recommended):**
   - On Windows (PowerShell):
     ```sh
     $env:GROQ_API_KEY="your-groq-api-key"
     $env:PINECONE_API_KEY="your-pinecone-api-key"
     ```
   - On macOS/Linux:
     ```sh
     export GROQ_API_KEY="your-groq-api-key"
     export PINECONE_API_KEY="your-pinecone-api-key"
     ```
   Or, edit the variables directly in `app.py` for demo/testing purposes.

## Running the Tech Advisor Chatbot

1. Launch the chatbot UI:
   ```sh
   streamlit run app.py
   ```
2. Ask your tech questions in the Streamlit interface!

## Customizing Knowledge Base

- The FAQ content is defined in `app.py` as `FAQ_CONTENT`.
- To add or update knowledge, edit the `FAQ_CONTENT` string or modify the document loading logic in `app.py`.

## Notes
- This demo uses Pinecone index `techadvisor` (512 dimensions, cosine metric, AWS us-east-1).
- Embeddings are generated using Sentence Transformers (locally) and answers are generated using Groq's Llama model.
- For production, always keep your API keys secure (use environment variables).
- The chatbot answers only from the provided FAQ context.
