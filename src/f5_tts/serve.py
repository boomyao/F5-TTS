from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from f5_tts.api import F5TTS
import random
import sys
import soundfile as sf
import tempfile
from f5_tts.tools.oss import upload_file, cache_file
from f5_tts.tools.enhance import enhance_audio
from uuid import uuid4
import torch
app = FastAPI()

f5tts = F5TTS(vocoder_name="bigvgan")

@app.post("/tts")
async def tts(request: Request):
    data = await request.json()
    text = data.get("text")
    ref_object_name = data.get("ref_object_name", None)
    ref_text = data.get("ref_text", None)
    speed = data.get("speed", 1.0)
    should_enhance = data.get("should_enhance", False)

    if text is None or ref_object_name is None or ref_text is None:
        return JSONResponse(status_code=400, content={"error": "text, ref_object_name and ref_text cannot be all provided"})

    ref_file = cache_file(ref_object_name)
    print(ref_file)
    seed = data.get("seed")

    if seed is None:
        seed = random.randint(0, sys.maxsize)

    print(f"Generating audio for {text} with seed {seed}")
    wav, sr, spect = f5tts.infer(gen_text=text, ref_file=ref_file, ref_text=ref_text, seed=seed, speed=float(speed))

    if should_enhance:
        try:
            wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav, sr = enhance_audio(wav, sr)
            wav = wav.numpy()
        finally:
            torch.cuda.empty_cache()

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, wav, sr)
        object_name = f"zero_shot/{uuid4()}.wav"
        upload_file(f.name, object_name)
    
    return JSONResponse(content={"object_name": object_name, "seed": seed})

def main():
    uvicorn.run(app, host="0.0.0.0", port=6200)