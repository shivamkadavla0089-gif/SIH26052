import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference.enhance import engine

app = FastAPI(title="SIH Speech Enhancement Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "Battlefield AI Speech Enhancement"}


@app.post("/api/enhance")
async def enhance_audio(
    file: UploadFile = File(...),
    noise_type: str = Form("battlefield")
):
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file.")
        result = engine.enhance(contents, noise_profile=noise_type)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)