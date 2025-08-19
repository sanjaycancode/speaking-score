from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
from PTEScorer import SpeakingScorer

app = FastAPI()

# Allow all origins, methods, and headers
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # allow all methods (GET, POST, etc.)
#     allow_headers=["*"],  # allow all headers
# )


@app.get("/")
def read_root():
    return {"Hello": "World"}

class ReadAloud(BaseModel):
    audio_url: Union[str, None] = None
    reference_text: Union[str, None] = None

@app.post("/speaking/read-aloud")
async def score_speaking(ReadAloud: ReadAloud):
    audio_url = ReadAloud.audio_url
    reference_text = ReadAloud.reference_text

    if not audio_url or not reference_text:
        return {"message": "Missing audio_url or reference_text", "status": 400}

    scorer = SpeakingScorer("base")
    result = scorer.score_speaking_task(audio_url, reference_text, "read_aloud")

    return result
 
    