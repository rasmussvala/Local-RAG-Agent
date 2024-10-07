# Local RAG Agent

A local RAG agent you can add documents to and ask questions about. Still a work in progress.

## Installation

1. Clone repo with LLM: `git clone --recurse-submodules https://github.com/rasmussvala/Local-RAG-Agent.git`
2. Create virtual environment: `py -m venv .venv`
3. Activate virtual environment (Windows): `.\.venv\Scripts\activate`
4. Install requirements:
   - Without CUDA: `pip install -r .\requirements.txt`
   - With CUDA: `pip install -r .\requirements_cuda.txt`

## Usage

### Adding documents

To add documents to the chatbot you need to do the following:

1. Replace example documents in the **documents** folder (right now .txt is only supported).
2. Run the **proccess documents** script to embed documents so chatbot can find them.

```
py .\proccess_documents.py
```

### Chatting

To chat with the chatbot you run the script

```
py .\main.py
```

The chatbot as of now only finds relevant documents in the first query. If you want the chatbot to find new documents you new to start a new session.

## Future improvements

- Create a command "restart" to restart chat
- GUI
