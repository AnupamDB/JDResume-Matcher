from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tkinter as tk
from tkinter import filedialog
from matcher import match_resumes
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def pick_folder():
    # Use tkinter to open a folder dialog natively
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path

@app.get("/api/select-folder")
def select_folder():
    path = pick_folder()
    return {"path": path}

class MatchRequest(BaseModel):
    jd_folder: str
    resume_folder: str
    output_folder: str
    top_k: int = 5

@app.post("/api/match")
def match(req: MatchRequest):
    try:
        match_resumes(req.jd_folder, req.resume_folder, req.output_folder, req.top_k)
        return {"status": "success", "message": "Matching complete!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
