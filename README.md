# Local-RAG-Agent

A local RAG agent you can add documents to and ask questions about. Still a work in progress.

## Future improvements

- Create a command "clear" to restart chat
- GUI
- Easier document add

# Run it on your machine

1. Clone repo with LLM: `git clone --recurse-submodules https://github.com/rasmussvala/Local-RAG-Agent.git`
2. Create virtual environment: `py -m venv .venv`
3. Activate virtual environment (Windows): `.\.venv\Scripts\activate`
4. Install requirements:
   - Without CUDA: `pip install -r .\requirements.txt`
   - With CUDA: `pip install -r .\requirements_cuda.txt`
