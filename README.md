# Local-RAG-Agent

A local RAG agent that you can add documents to and ask questions about. Very much still a work in progress.

## Notes to self

- Currently pytorch is installed without cuda support, it's very slow, maybe uninstall, install with cuda and make it only usable for cuda sopported gpus
- create a command "clear" to restart chat

# Run it on your machine

1. Clone repo with LLM: `git clone --recurse-submodules https://github.com/rasmussvala/Local-RAG-Agent.git`
2. Create venv: `py -m venv .venv`
3. Activate venv (Windows): `.\.venv\Scripts\activate`
4. Install requirments:
   - Without CUDA: `pip install -r .\requirements.txt`
   - With CUDA: `pip install -r .\requirements_cuda.txt`
