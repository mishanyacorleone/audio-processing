import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api import api
from app.config.config import OUTPUT_DIR


app = FastAPI(title="Audio Processing API", version="1.0")

app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

app.include_router(api.router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Audio Processing API. Документация: /docs"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)