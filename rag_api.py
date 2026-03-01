from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from fastapi.middleware.cors import CORSMiddleware

from backend2 import (
    workflow,
    reterive_all_threads,
    delete_thread
)

from langchain_core.messages import HumanMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    thread_id: str
    user_id: str
    


@app.post("/chat")
async def chat(req: ChatRequest):

    CONFIG = {
        "configurable": {
            "thread_id": req.thread_id,
            "user_id": req.user_id
        }
    }

    initial_input = {
        "question": req.question,
        "messages": [HumanMessage(content=req.question)],
        "docs": [],
        "good_docs": [],
        "web_docs": [],
        "refined_context_docs": []
    }

    async def event_generator():

        for update in workflow.stream(
            initial_input,
            config=CONFIG,
            stream_mode="updates"
        ):

            for node, values in update.items():

                payload = {"node": node}

                if "answer" in values:
                    payload["answer"] = values["answer"]

                if "verdict" in values:
                    payload["verdict"] = values["verdict"]
                    payload["reason"] = values.get("reason")

                if "web_query" in values:
                    payload["web_query"] = values["web_query"]

                yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/threads")
def get_threads(user_id: str):
    return reterive_all_threads(user_id)




@app.delete("/threads/{thread_id}")
def remove_thread(thread_id: str):
    ok = delete_thread(thread_id)
    return {"success": ok}




@app.get("/threads/{thread_id}/messages")
def load_messages(thread_id: str):

    state = workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    messages = []

    for msg in state.values.get("messages", []):
        role = "assistant"
        if msg.type == "human":
            role = "user"

        messages.append({
            "role": role,
            "content": msg.content
        })

    return messages




from backend2 import MultiRag
import os

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):

    os.makedirs("uploads_doc", exist_ok=True)

    paths = []

    for f in files:
        path = f"uploads_doc/{f.filename}"
        with open(path, "wb") as buffer:
            buffer.write(await f.read())
        paths.append(path)

    rag = MultiRag()

    rag.build_enhanced_vector_database(
        file_paths=paths
    )

    return {"success": True, "message": "Files indexed"}


