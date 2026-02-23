# ContextFlow

A Django web application for uploading PDF documents and chatting with them using a RAG pipeline built on OpenAI and FAISS. Includes user authentication, persistent conversation history, cross-encoder reranking, input moderation, and prompt injection detection.

## Setup

1. Clone the repo and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the project root:
```
SECRET_KEY=your_django_secret_key
DEBUG=TRUE
OPENAI_API_KEY=your_openai_api_key
```
4. Apply migrations: `python manage.py migrate`
5. Collect static files: `python manage.py collectstatic`
6. Start the server: `python manage.py runserver`

## Running Tests

```bash
python manage.py test
```

## Notes

- PDF uploads are stored under `media/uploads/pdfs/`
- FAISS indexes are persisted under `faiss_store/` and reloaded on subsequent requests to avoid re-embedding
- Max upload size is 50MB per file and 200MB total per request
- The cross-encoder reranker runs on CPU by default