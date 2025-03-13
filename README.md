# golden-retriever

A simple TUI application for a more transparent, on-device RAG pipeline.

https://github.com/user-attachments/assets/e8c68937-8c12-4cdf-8ed2-1ec47193e98b

# To Run

1. Install dependencies. ```pip install -r requirements.txt```
2. Configure your OpenAI endpoints and models.
```
export OPENAI_BASE_URL=http://127.0.0.1:1234/v1
export OPENAI_API_KEY=your_api_key_if_needed
export GR_LLM_MODEL=mistral-nemo-instruct-2407

```
2.5. Ensure your servers are running if you're hosting locally.
3. Run ```python src/gui.py```
