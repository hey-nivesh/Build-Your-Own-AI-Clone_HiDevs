@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Cleaning up conflicting packages...
pip uninstall pinecone-client pinecone -y

echo Installing required packages...
pip install streamlit langchain sentence-transformers pinecone-client requests

echo Setup complete!
pause