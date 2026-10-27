"""Local HTTP API exposing the RAG pipeline (bonus part)."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.chunking import FILES_DIR
from src.chat import append_exchange, delete_chat, get_chat, list_chats
from src.logger import FatalError
from src.rag import RAG, CHUNKS_FILE

ALLOWED_SUFFIXES = (".pdf", ".docx", ".md", ".txt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG()


class DeleteFileRequest(BaseModel):
    file_path: str


class RenameDocumentRequest(BaseModel):
    original_name: str


@app.get("/status")
def status() -> dict[str, int]:
    """Health check."""
    return {"status": 200}


@app.get("/")
def root() -> dict[str, str]:
    """Welcome message."""
    return {"value": "Hello world from rag against the machine project!"}


def _rebuild_index(max_chunk_size: int = 2000) -> None:
    rag.index(max_chunk_size)


def _save_upload(file: UploadFile) -> tuple[Path, str]:
    """Save an uploaded file under data/files and return its path and name."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name")

    original = Path(file.filename).name
    if not original.lower().endswith(ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf, .docx, .md, .txt files are supported",
        )

    files_dir = Path(FILES_DIR)
    files_dir.mkdir(parents=True, exist_ok=True)
    out_path = files_dir / f"{uuid.uuid4().hex}_{original}"
    with open(out_path, "wb") as f:
        f.write(file.file.read())
    with open(f"{out_path}.meta.json", "w", encoding="utf-8") as f:
        json.dump({"original_name": original}, f)
    return out_path, original


def _document_path(document_id: str) -> Path:
    """Resolve a document id to a file inside data/files."""
    if Path(document_id).name != document_id:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(FILES_DIR) / document_id
    if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=404, detail="Document not found")
    return path


def _chunks() -> list:
    """Return the persisted chunks, or an empty list."""
    try:
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _chunk_count(path: Path) -> int:
    """Count the indexed chunks that belong to ``path``."""
    return sum(
        1 for c in _chunks()
        if isinstance(c, dict) and c.get("file_path") == str(path)
    )


def _original_name(document_id: str) -> str:
    """Recover the original file name from metadata or the document id."""
    meta = Path(FILES_DIR) / f"{document_id}.meta.json"
    try:
        with open(meta, encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("original_name")
        if isinstance(name, str) and name:
            return name
    except (OSError, json.JSONDecodeError):
        pass
    prefix, sep, rest = document_id.partition("_")
    if sep and len(prefix) == 32:
        return rest
    return document_id


@app.post("/index")
def index(
    max_chunk_size: int = Form(2000),
    file: Optional[UploadFile] = File(None),
) -> dict[str, str]:
    """(Re)build the corpus index.

    If a file is uploaded, it is saved into data/files before indexing.
    """
    try:
        if file is not None:
            _save_upload(file)

        _rebuild_index(max_chunk_size)
    except HTTPException:
        raise
    except (FatalError, OSError, ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail="Error encountered while indexing...",
        ) from e
    return {"status": "Successfully indexed!"}


@app.post("/upload")
def upload(
    max_chunk_size: int = Form(2000),
    file: UploadFile = File(...),
) -> dict:
    """Upload a file to data/files and rebuild index."""
    try:
        out_path, original = _save_upload(file)
        _rebuild_index(max_chunk_size)
    except HTTPException:
        raise
    except (FatalError, OSError, ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail="Error encountered while indexing...",
        ) from e
    return {
        "document_id": out_path.name,
        "original_name": original,
        "chunks": _chunk_count(out_path),
    }


@app.get("/documents")
def documents() -> dict:
    """List the uploaded documents."""
    counts: dict[str, int] = {}
    for chunk in _chunks():
        if isinstance(chunk, dict):
            path = chunk.get("file_path")
            if isinstance(path, str):
                counts[path] = counts.get(path, 0) + 1

    files_dir = Path(FILES_DIR)
    result = []
    if files_dir.is_dir():
        for path in sorted(files_dir.iterdir()):
            suffix = path.suffix.lower()
            if not path.is_file() or suffix not in ALLOWED_SUFFIXES:
                continue
            result.append({
                "document_id": path.name,
                "original_name": _original_name(path.name),
                "chunk_count": counts.get(str(path), 0),
                "uploaded_at": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat(),
            })
    return {"documents": result}


@app.get("/documents/{document_id}/content")
def document_content(document_id: str) -> dict:
    """Return the indexed text of one document."""
    path = _document_path(document_id)
    chunks = [
        c for c in _chunks()
        if isinstance(c, dict) and c.get("file_path") == str(path)
    ]
    chunks.sort(key=lambda c: c.get("first_character_index", 0))
    return {
        "document_id": document_id,
        "content": "".join(c.get("content", "") for c in chunks),
    }


@app.get("/documents/{document_id}/download")
def document_download(document_id: str) -> FileResponse:
    """Download the original uploaded file."""
    path = _document_path(document_id)
    return FileResponse(path, filename=_original_name(document_id))


@app.get("/documents/{document_id}/raw")
def document_raw(document_id: str) -> FileResponse:
    """Serve the original file inline so the browser can preview it."""
    return FileResponse(_document_path(document_id))


@app.patch("/documents/{document_id}")
def document_rename(
    document_id: str, req: RenameDocumentRequest
) -> dict:
    """Set the displayed name of an uploaded document."""
    _document_path(document_id)
    name = req.original_name.strip()
    if not name:
        raise HTTPException(
            status_code=400, detail="original_name must not be empty"
        )
    meta = Path(FILES_DIR) / f"{document_id}.meta.json"
    try:
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"original_name": name}, f)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail="Error encountered while renaming the document...",
        ) from e
    return {"document_id": document_id, "original_name": name}


@app.delete("/documents/{document_id}")
def document_delete(document_id: str) -> dict:
    """Delete a document and its chunks from the index."""
    path = _document_path(document_id)
    try:
        rag.delete_file(str(path))
        Path(f"{path}.meta.json").unlink(missing_ok=True)
    except (FatalError, OSError, ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail="Error encountered while deleting the document...",
        ) from e
    return {"status": "deleted"}


@app.delete("/files")
def delete_file(req: DeleteFileRequest) -> dict[str, str]:
    """Delete a file and remove its chunks from the processed index."""
    try:
        target = Path(req.file_path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            target.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        chunks_path = Path(CHUNKS_FILE)
        if chunks_path.exists():
            try:
                with open(chunks_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    filtered = [
                        c for c in data
                        if c.get("file_path") != str(target)
                        and c.get("file_path") != req.file_path
                    ]
                    with open(chunks_path, "w", encoding="utf-8") as f:
                        json.dump(filtered, f, indent=2)
            except Exception:
                pass

        try:
            rag.index(2000)
        except Exception:
            pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"status": "deleted"}


@app.get("/chats")
def chats() -> dict:
    return {"chats": list_chats()}


@app.get("/chats/{chat_id}")
def chat(chat_id: str) -> dict:
    result = get_chat(chat_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return result


@app.delete("/chats/{chat_id}")
def chat_del(chat_id: str) -> dict:
    delete_chat(chat_id)
    return {"status": "ok"}


@app.get("/search")
def search_endpoint(prompt: str = "", k: int = 5) -> dict[str, Any]:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be positive")
    try:
        sources_info = rag.get_sources_info(query=prompt, k=k)
        # Enrich with original names
        for s in sources_info:
            doc_id = s.get("document_id") or Path(s.get("file_path", "")).name
            try:
                s["original_name"] = _original_name(doc_id)
            except Exception:
                pass
        # Also return full chunk info if needed
        res = rag.search(prompt, k)
        chunks = []
        for s in res.retrieved_sources[:5]:
            chunks.append({
                "file_path": s.file_path,
                "document_id": Path(s.file_path).name,
                "content": s.content,
                "first_character_index": s.first_character_index,
                "last_character_index": s.last_character_index,
            })
        return {"sources": sources_info, "chunks": chunks}
    except (FatalError, OSError, ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail="Error encountered while searching...") from e


@app.get("/answer")
def answer(
    prompt: str = "", k: int = 5, chat_id: str = ""
) -> dict[str, Any]:
    """Answer a question using the RAG pipeline and save the exchange."""
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be positive")
    history: list[dict[str, Any]] = []
    if chat_id:
        existing = get_chat(chat_id)
        if existing:
            messages = existing.get("messages", [])
            if isinstance(messages, list):
                history = messages
    try:
        sources_info = rag.get_sources_info(query=prompt, k=k)
        for s in sources_info:
            doc_id = s.get("document_id") or Path(s.get("file_path", "")).name
            try:
                s["original_name"] = _original_name(doc_id)
            except Exception:
                pass
        text = rag.answer(query=prompt, k=k, history=history)
        chat_id = append_exchange(chat_id, prompt, text, sources=sources_info)
    except (FatalError, OSError, ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail="Error encountered while answering...",
        ) from e
    return {"answer": text, "chat_id": chat_id, "sources": sources_info}
