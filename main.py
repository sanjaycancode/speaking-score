
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PTEScorer import SpeakingScorer
from util.api_helper import async_api_handler

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
    async def process_request():
        if not audio_file or not reference_text:
            return {"error": "Missing audio file or reference text", "status": 400}
        import tempfile
        import shutil
        import os
        ext = os.path.splitext(audio_file.filename)[1].lower()
        if ext not in [".mp3", ".wav"]:
            ext = ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            audio_path = tmp.name
        scorer = SpeakingScorer("base")
        result = scorer.score_speaking_task(audio_path, reference_text, "read_aloud")
        return result

    return await async_api_handler(process_request)
 
    