# StatArb Frontend

React + Vite + TypeScript + Tailwind dashboard for the StatArb pipeline.
Theme: dark navy + white. The Streamlit app is still available — this UI
sits alongside it.

## Run

Two processes — start them in separate terminals.

### 1. Backend (FastAPI, wraps the existing `statarb` modules)

```bash
# from repo root
pip install -r backend/requirements.txt   # one-time
uvicorn backend.app:app --reload --port 8000
```

The backend imports `config.py` and `statarb.*` directly — no Python
logic is duplicated.

### 2. Frontend (Vite dev server)

```bash
cd frontend
npm install        # one-time
npm run dev        # serves on http://localhost:5173
```

Vite proxies `/api/*` to `http://localhost:8000`, so the same code works
in dev and behind a reverse proxy in prod.

## Build for production

```bash
cd frontend
npm run build      # outputs to frontend/dist
```

Serve `frontend/dist` with any static host (nginx, Cloudflare Pages,
etc.) and point `/api/*` at the FastAPI service.
