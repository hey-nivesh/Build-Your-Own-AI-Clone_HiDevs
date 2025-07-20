import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from pinecone import Pinecone, ServerlessSpec
import time
from sentence_transformers import SentenceTransformer
import tempfile
import numpy as np

# --- Configuration ---
GROQ_API_KEY = "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"
PINECONE_API_KEY = "pcsk_hBpNB_KKfcZ9foCoqnDUJfCVCMzsgtpXXcEdrS6WoaT5owoQ7DU1gfiPmRcNahT2EzjSJ"
INDEX_NAME = "techadvisor"
EMBEDDING_DIMENSION = 512  # Match your existing index
PINECONE_ENVIRONMENT = "us-east-1"  # From your index details

# --- Initialize session state ---
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'pinecone_client' not in st.session_state:
    st.session_state.pinecone_client = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Create FAQ content ---
FAQ_CONTENT = """Tech Advisor FAQ

What is Python?
Python is a popular programming language known for its readability and versatility. It's widely used for web development, data analysis, artificial intelligence, automation, and scientific computing. Python's syntax is designed to be intuitive and its extensive library ecosystem makes it suitable for beginners and experts alike.

How do I install Node.js?
You can download Node.js from the official website (nodejs.org) and follow the installation instructions for your operating system. For Windows, download the .msi installer and run it. For macOS, you can use the .pkg installer or install via Homebrew with 'brew install node'. For Linux, you can use your distribution's package manager or download from the official site.

What is a virtual environment?
A virtual environment is an isolated Python environment that allows you to manage dependencies for different projects separately. It prevents conflicts between different versions of packages. You can create one using 'python -m venv myenv' and activate it with 'myenv\\Scripts\\activate' on Windows or 'source myenv/bin/activate' on macOS/Linux.

What is Git?
Git is a distributed version control system that tracks changes in source code during software development. It allows multiple developers to work on the same project simultaneously and keeps a complete history of all changes. Common Git commands include 'git init', 'git add', 'git commit', 'git push', and 'git pull'.

How do I debug Python code?
There are several ways to debug Python code: 1) Use print statements to output variable values, 2) Use the built-in debugger (pdb) by adding 'import pdb; pdb.set_trace()', 3) Use IDE debugging features in PyCharm, VSCode, or other editors, 4) Use logging instead of print for better control, 5) Write unit tests to catch bugs early.

What is an API?
An API (Application Programming Interface) is a set of protocols, routines, and tools that allow different software applications to communicate with each other. APIs define how software components should interact and are used to enable integration between different systems. REST APIs are commonly used for web services.

How do I handle errors in Python?
Python uses try-except blocks for error handling. You can catch specific exceptions or use a general except clause. Best practices include: catching specific exceptions rather than using bare except, using finally blocks for cleanup code, raising custom exceptions when appropriate, and logging errors for debugging purposes.

What is machine learning?
Machine learning is a subset of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, without being explicitly programmed for each task. Common types include supervised learning (learning from labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error).

How do I optimize Python code performance?
To optimize Python code: 1) Use built-in functions and libraries (they're implemented in C), 2) Avoid unnecessary loops and use list comprehensions, 3) Use appropriate data structures (sets for membership testing, dictionaries for lookups), 4) Profile your code to identify bottlenecks, 5) Consider using NumPy for numerical computations, 6) Use caching for expensive operations.

What is Docker?
Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers. Containers ensure that applications run consistently across different environments. Docker helps with deployment, scaling, and managing applications in development, testing, and production environments.

How do I secure a web application?
Web application security involves multiple layers: 1) Use HTTPS for all communications, 2) Implement proper authentication and authorization, 3) Validate and sanitize all user inputs, 4) Protect against SQL injection and XSS attacks, 5) Keep dependencies updated, 6) Implement rate limiting, 7) Use security headers, 8) Regular security audits and penetration testing.

What is cloud computing?
Cloud computing delivers computing services (servers, storage, databases, networking, software) over the internet. Main service models include Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). Benefits include scalability, cost-effectiveness, accessibility, and reduced maintenance overhead."""

# --- 1. Load embedding model ---
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model"""
    try:
        # Using a model that can be adjusted to 512 dimensions
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim base
        st.success("✅ Embedding model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model: {str(e)}")
        return None

# --- 2. Setup Pinecone ---
@st.cache_resource
def setup_pinecone():
    """Initialize Pinecone client and connect to existing index"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            st.warning(f"⚠️ Index '{INDEX_NAME}' not found. Creating new index...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=PINECONE_ENVIRONMENT
                )
            )
            # Wait for index to be ready
            time.sleep(10)
            st.success(f"✅ Index {INDEX_NAME} created successfully!")
        else:
            st.info(f"🔗 Connected to existing Pinecone index: {INDEX_NAME}")
        
        # Connect to index
        index = pc.Index(INDEX_NAME)
        
        # Get index stats
        stats = index.describe_index_stats()
        st.info(f"📊 Index stats: {stats.total_vector_count} vectors")
        
        return pc, index
        
    except Exception as e:
        st.error(f"❌ Error setting up Pinecone: {str(e)}")
        return None, None

# --- 3. Enhanced embedding function ---
def get_embedding(text, model):
    """Generate embedding with proper dimension handling"""
    try:
        # Generate base embedding
        embedding = model.encode(text)
        
        # Convert to proper dimensions (512) by padding/truncating
        if len(embedding) < EMBEDDING_DIMENSION:
            # Pad with zeros
            padded_embedding = np.pad(embedding, (0, EMBEDDING_DIMENSION - len(embedding)), 'constant')
            return padded_embedding.tolist()
        elif len(embedding) > EMBEDDING_DIMENSION:
            # Truncate
            return embedding[:EMBEDDING_DIMENSION].tolist()
        else:
            return embedding.tolist()
            
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

# --- 4. Load and chunk documents ---
def load_and_chunk_docs():
    """Load and chunk the FAQ document"""
    try:
        # Create temporary file with FAQ content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(FAQ_CONTENT)
            tmp_file_path = tmp_file.name
        
        # Load document
        loader = TextLoader(tmp_file_path, encoding='utf-8')
        docs = loader.load()
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks for better retrieval
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        
        # Extract text content from chunks
        texts = [chunk.page_content.strip() for chunk in chunks if chunk.page_content.strip()]
        
        st.info(f"📄 Loaded {len(texts)} chunks from FAQ document")
        return texts
        
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")
        return []

# --- 5. Store chunks in Pinecone ---
def store_chunks(index, texts, model):
    """Store text chunks as vectors in Pinecone"""
    try:
        # Check if vectors already exist
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            st.info(f"📊 Index already contains {stats.total_vector_count} vectors. Skipping upload.")
            return True
        
        vectors = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            
            # Generate embedding
            embedding = get_embedding(text, model)
            if embedding is None:
                continue
            
            vector = {
                "id": f"chunk-{i}",
                "values": embedding,
                "metadata": {
                    "text": text[:1000],  # Limit metadata text length
                    "chunk_id": i,
                    "source": "FAQ"
                }
            }
            vectors.append(vector)
            
            # Update progress
            progress_bar.progress((i + 1) / len(texts))
        
        # Upsert vectors in batches
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            time.sleep(0.1)  # Small delay to avoid rate limits
        
        progress_bar.empty()
        st.success(f"✅ Stored {len(vectors)} text chunks in Pinecone!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error storing chunks: {str(e)}")
        return False

# --- 6. Enhanced retrieval function ---
def retrieve_chunks(index, query, model, top_k=5):
    """Retrieve most relevant chunks for the query"""
    try:
        # Generate query embedding
        query_embedding = get_embedding(query, model)
        if query_embedding is None:
            return []
        
        # Query Pinecone with filters
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"source": "FAQ"}  # Filter to only FAQ content
        )
        
        if not results or not results.matches:
            return []
        
        # Extract text from results with scores
        retrieved_texts = []
        for match in results.matches:
            if match.metadata and 'text' in match.metadata:
                # Only include results with good similarity scores
                if match.score > 0.7:  # Adjust threshold as needed
                    retrieved_texts.append({
                        'text': match.metadata['text'],
                        'score': match.score
                    })
        
        st.info(f"🔍 Retrieved {len(retrieved_texts)} relevant chunks")
        return retrieved_texts
        
    except Exception as e:
        st.error(f"❌ Error retrieving chunks: {str(e)}")
        return []

# --- 7. Enhanced answer generation ---
def generate_answer(context, question):
    """Generate answer using Groq's Llama model"""
    if not context:
        return "I couldn't find relevant information to answer your question. Please try asking about Python, Node.js, Git, Docker, or other tech topics from our FAQ."
    
    # Prepare context from retrieved chunks
    context_text = "\n\n".join([item['text'] if isinstance(item, dict) else item for item in context])
    
    prompt = f"""You are a friendly and knowledgeable tech advisor AI assistant. 
Answer the user's question using ONLY the context provided below. 
Be helpful, accurate, and conversational. If the context doesn't contain enough information to fully answer the question, say so politely and suggest what topics you can help with.

Context:
{context_text}

Question: {question}

Answer:"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful tech advisor AI. Answer questions based only on the provided context. Be conversational and helpful."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": 600,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", 
            json=data, 
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            st.error(f"Groq API Error: {response.status_code}")
            return "Sorry, I encountered an error while generating the answer. Please try again."
            
    except requests.exceptions.Timeout:
        return "Sorry, the request timed out. Please try again."
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return "Sorry, I couldn't generate an answer at the moment."

# --- 8. Initialize system ---
def initialize_system():
    """Initialize the RAG system components"""
    if st.session_state.embedding_model is None:
        with st.spinner("🔄 Loading embedding model..."):
            st.session_state.embedding_model = load_embedding_model()
    
    if st.session_state.pinecone_client is None or st.session_state.index is None:
        with st.spinner("🔄 Connecting to Pinecone..."):
            pc, index = setup_pinecone()
            st.session_state.pinecone_client = pc
            st.session_state.index = index
    
    # Load documents if not already loaded
    if not st.session_state.documents_loaded and st.session_state.index is not None and st.session_state.embedding_model is not None:
        with st.spinner("🔄 Loading and storing FAQ documents..."):
            texts = load_and_chunk_docs()
            if texts:
                success = store_chunks(st.session_state.index, texts, st.session_state.embedding_model)
                st.session_state.documents_loaded = success

# --- 9. Main query function ---
def answer_query(user_query):
    """Main RAG pipeline function"""
    try:
        if not st.session_state.embedding_model or not st.session_state.index:
            return "Error: System not properly initialized", []
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_chunks(
            st.session_state.index, 
            user_query, 
            st.session_state.embedding_model
        )
        
        if not relevant_chunks:
            return "I couldn't find relevant information for your question. Try asking about Python, Node.js, Git, Docker, APIs, or other programming topics!", []
        
        # Generate answer
        answer = generate_answer(relevant_chunks, user_query)
        return answer, relevant_chunks
        
    except Exception as e:
        st.error(f"❌ Error in RAG pipeline: {str(e)}")
        return f"Error: {str(e)}", []

# --- 10. Enhanced Streamlit UI ---
def main():
    st.set_page_config(
        page_title="Tech Advisor AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Tech Advisor AI")
    st.markdown("Ask me any technical questions and I'll help you based on our comprehensive FAQ knowledge base!")
    
    # Initialize system
    initialize_system()
    
    # Sidebar with enhanced information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This RAG-based chatbot uses:
        - **🧠 Sentence Transformers** for embeddings
        - **📊 Pinecone** for vector storage  
        - **⚡ Groq (Llama 3)** for answer generation
        - **🔗 Langchain** for document processing
        """)
        
        # System status with enhanced indicators
        st.header("🔧 System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.embedding_model:
                st.success("✅ Embeddings")
            else:
                st.error("❌ Embeddings")
                
        with col2:
            if st.session_state.index:
                st.success("✅ Pinecone")
            else:
                st.error("❌ Pinecone")
        
        if st.session_state.documents_loaded:
            st.success("✅ Knowledge Base Loaded")
        else:
            st.warning("⚠️ Loading Knowledge Base...")
        
        # Index information
        if st.session_state.index:
            try:
                stats = st.session_state.index.describe_index_stats()
                st.metric("📊 Vector Count", stats.total_vector_count)
            except:
                pass
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "What is Python?",
            "How do I install Node.js?", 
            "What is a virtual environment?",
            "How do I use Git?",
            "What is Docker?",
            "How do I debug Python code?",
            "What is machine learning?",
            "How do I secure a web application?",
            "What is cloud computing?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.user_input = question
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat input
        user_input = st.text_input(
            "Ask a tech question:", 
            value=st.session_state.get('user_input', ''),
            placeholder="e.g., What is Python and why should I learn it?",
            key="main_input"
        )
    
    with col2:
        ask_button = st.button("Ask 🚀", use_container_width=True)
    
    # Clear the session state after using it
    if 'user_input' in st.session_state:
        del st.session_state.user_input
    
    # Process query
    if user_input and (ask_button or user_input != st.session_state.get('last_query', '')):
        st.session_state.last_query = user_input
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ System is still loading. Please wait a moment and try again.")
        else:
            with st.spinner("🔍 Searching knowledge base and generating answer..."):
                answer, docs = answer_query(user_input)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': user_input,
                'answer': answer,
                'chunks': docs
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 chats
            with st.expander(f"Q: {chat['question'][:60]}..." if len(chat['question']) > 60 else f"Q: {chat['question']}", expanded=(i==0)):
                st.markdown("**Answer:**")
                st.write(chat['answer'])
                
                # Show retrieved context
                if chat['chunks']:
                    with st.expander("📚 Retrieved Context"):
                        for j, chunk in enumerate(chat['chunks'][:3]):  # Show top 3 chunks
                            if isinstance(chunk, dict):
                                st.text_area(
                                    f"Context {j+1} (Score: {chunk.get('score', 'N/A'):.3f}):",
                                    chunk['text'],
                                    height=100,
                                    disabled=True,
                                    key=f"context_{i}_{j}_{hash(str(chunk))}"
                                )
                            else:
                                st.text_area(
                                    f"Context {j+1}:",
                                    chunk,
                                    height=100,
                                    disabled=True,
                                    key=f"context_{i}_{j}_{hash(chunk)}"
                                )

if __name__ == "__main__":
    main()
