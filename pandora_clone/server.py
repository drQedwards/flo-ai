from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import model

app = FastAPI(title="Pandora GPT-5o (Toy)")


class ChatCompletionRequest(BaseModel):
    # In a real implementation, messages would be richer objects.
    # For this toy demo we accept a 2-D list of floats representing token embeddings.
    embeddings: list[list[float]] = Field(..., description="2D array [seq_len][d_model]")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    out = model.run(req.embeddings)
    return {
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": str(out),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": len(req.embeddings), "completion_tokens": len(out)},
    }


# Development entry point: uvicorn pandora_clone.server:app --reload