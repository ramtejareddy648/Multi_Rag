from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from fastapi import Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
# from backend2 import MultiRag
from dotenv import load_dotenv
load_dotenv()

from backend2 import workflow_fun, reterive_all_threads, delete_thread

from langchain_core.messages import HumanMessage

app = FastAPI()


workflow_instance = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global workflow_instance
    try:
        print("Initializing workflow...")
        workflow_instance = workflow_fun()
        print("Workflow ready ✅")
    except Exception as e:
        print(f"Startup failed ❌: {e}")
        workflow_instance = None
        
        
        
def get_workflow():
    return workflow_instance



@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    return Response(status_code=200)

class ChatRequest(BaseModel):
    question: str
    thread_id: str
    user_id: str
    


@app.post("/chat",response_class=StreamingResponse)
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
        
        workflow = get_workflow()

        if workflow is None:
            yield f"data: {json.dumps({'error': 'Workflow not initialized'})}\n\n"
            return

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
    
    workflow = get_workflow()
    if workflow is None:
        return {"error": "Workflow not initialized"}
    return reterive_all_threads(workflow, user_id)
    # return reterive_all_threads(user_id)




@app.delete("/threads/{thread_id}")
def remove_thread(thread_id: str):
    
    workflow = get_workflow()
    if workflow is None:
        return {"error": "Workflow not initialized"}
    ok = delete_thread(workflow, thread_id)
    return {"success": ok}




@app.get("/threads/{thread_id}/messages")
def load_messages(thread_id: str):
    
    
    workflow = get_workflow()
    
    if workflow is None:
        return {"error": "Workflow not initialized"}

    

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






@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):

    os.makedirs("uploads_doc", exist_ok=True)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    doc_paths = []
    img_paths = []
    

   

    for f in files:
        path = f"uploads_doc/{f.filename}"
        with open(path, "wb") as buffer:
            buffer.write(await f.read())
        
        file_ext = os.path.splitext(f.filename)[1].lower()
        if file_ext in image_extensions:
            img_paths.append(path)
        else:
            doc_paths.append(path)
        
        

    workflow = get_workflow()
    
    if workflow is None:
        return {"error": "Workflow not initialized"}
    rag = workflow._rag

    rag.build_enhanced_vector_database(
        image_paths=img_paths,
        file_paths=doc_paths
    )

    return {"success": True, "message": "Files indexed"}




@app.post("/speech-to-text")
async def speech_to_text_api(file: UploadFile = File(...)):

    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

    if not SARVAM_API_KEY:
        return {"error": "SARVAM_API_KEY not set"}

    audio_bytes = await file.read()

    url = "https://api.sarvam.ai/speech-to-text"

    files = {
        "file": ("audio.wav", audio_bytes, "audio/wav")
    }

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    data = {
        "model": "saaras:v3",
        "mode": "translate",
        "language_code": "te-IN"
    }

    response = requests.post(
        url,
        headers=headers,
        files=files,
        data=data
    )

    if response.status_code != 200:
        return {"error": response.text}

    return {
        "transcript": response.json().get("transcript", "")
    }
    
    
    
    
    
    
@app.post("/text-to-speech")
async def text_to_speech_api(payload: dict):

    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

    if not SARVAM_API_KEY:
        return {"error": "SARVAM_API_KEY not set"}

    text = payload.get("text", "")

    url = "https://api.sarvam.ai/text-to-speech"

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "text": text[:500],
        "model": "bulbul:v3",
        "speaker": "shubh",
        "target_language_code": "en-IN"
    }

    response = requests.post(
        url,
        headers=headers,
        json=body
    )

    if response.status_code != 200:
        return {"error": response.text}

    audio_base64 = response.json()["audios"][0]

    return {
        "audio": audio_base64
    }
    
    
    
    
@app.get("/")
def health():
    if workflow_instance is None:
        return {"status": "error", "workflow": "not_ready"}
    return {"status": "ok", "workflow": "ready"}




@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

