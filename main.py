from typing import Union
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PTEScorer import SpeakingScorer

app = FastAPI()

# Allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/speaking/read-aloud")
async def score_speaking(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...)
):
    if not audio_file or not reference_text:
        return {"message": "Missing audio file or reference text", "status": 400}

    import tempfile
    import shutil
    import os
    
    try:
        ext = os.path.splitext(audio_file.filename)[1].lower()
        if ext not in [".mp3", ".wav"]:
            ext = ".mp3"  # default to mp3 if unknown
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            audio_path = tmp.name
    except Exception as e:
        return {"message": f"Error saving file: {e}", "status": 500}

    scorer = SpeakingScorer("base")
    try:
        result = scorer.score_speaking_task(audio_path, reference_text, "read_aloud")
    except Exception as e:
        return {"message": f"Error scoring: {e}", "status": 500}

    return result
 
    