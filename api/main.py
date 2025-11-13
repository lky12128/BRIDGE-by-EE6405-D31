import uvicorn
import time
import json
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

from model_loader import load_model, run_my_model_inference, GlobalModel

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    stream: Optional[bool] = False

app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_model()

async def stream_generator(prompt: str, model_name: str):
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())

    try:
        full_response = run_my_model_inference(prompt)
    except Exception as e:
        print(f"模型推理失败: {e}")
        full_response = "抱歉，我的模型在处理时遇到了一个错误。"

    words = full_response.split(" ")
    
    for word in words:
        if not word:
            continue
            
        chunk_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_timestamp,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None
                }
            ]
        }
        
        yield f"data: {json.dumps(chunk_data)}\n\n"
        
        await asyncio.sleep(0.05)

    stop_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_timestamp,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    user_prompt = ""
    if request.messages:
        user_prompt = request.messages[-1].content
    
    if not user_prompt:
        return {"error": "No prompt provided"}

    if request.stream:
        return StreamingResponse(
            stream_generator(user_prompt, request.model),
            media_type="text/event-stream"
        )
    else:
        full_response = run_my_model_inference(user_prompt)
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_response
                    },
                    "finish_reason": "stop"
                }
            ]
        }

@app.get("/")
def read_root():
    return {"message": "你的 BRIDGE 模型 API 服务器正在运行！"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
