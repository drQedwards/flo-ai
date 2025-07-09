from fastapi import FastAPI
from pydantic import BaseModel, Field

# Local imports
from . import model
from . import llm

app = FastAPI(title="Pandora GPT-5o (Toy)")


class ChatMessage(BaseModel):
    role: str
    content: str

# Update request model to support two modes
class ChatCompletionRequest(BaseModel):
    # Either embeddings vector input (legacy) OR messages (OpenAI style). If both provided, messages wins.
    embeddings: list[list[float]] | None = Field(None, description="2D array [seq_len][d_model]")
    messages: list[ChatMessage] | None = None
    max_tokens: int | None = 128
    temperature: float | None = 0.7


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.messages:
        # Build prompt from messages. Simple concatenation.
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant:"
        completion = llm.generate(prompt, max_tokens=req.max_tokens or 128, temperature=req.temperature or 0.7)
        return {
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
    elif req.embeddings:
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
    else:
        return {"error": "Provide either embeddings or messages."}


# Development entry point: uvicorn pandora_clone.server:app --reload