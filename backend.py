from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from matcher import match_resumes

import os
import shutil
import uuid


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# CONFIG
# ==============================

TEMP_ROOT = "temp_results"


# ==============================
# MATCH + ZIP
# ==============================

@app.post("/api/match")
async def match(
    jd_files: list[UploadFile] = File(...),
    resume_files: list[UploadFile] = File(...),
    top_k: int = Form(5)
):

    run_id = str(uuid.uuid4())

    base_dir = os.path.join(
        TEMP_ROOT,
        run_id
    )

    jd_folder = os.path.join(
        base_dir,
        "JDs"
    )

    resume_folder = os.path.join(
        base_dir,
        "Resumes"
    )

    output_folder = os.path.join(
        base_dir,
        "Matched_Resumes"
    )

    try:

        # ==============================
        # CREATE DIRECTORIES
        # ==============================

        os.makedirs(jd_folder, exist_ok=True)
        os.makedirs(resume_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)


        # ==============================
        # SAVE JD FILES
        # ==============================

        for file in jd_files:

            if not file.filename:
                continue

            filename = os.path.basename(file.filename)

            file_path = os.path.join(
                jd_folder,
                filename
            )

            with open(file_path, "wb") as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )


        # ==============================
        # SAVE RESUME FILES
        # ==============================

        for file in resume_files:

            if not file.filename:
                continue

            filename = os.path.basename(file.filename)

            file_path = os.path.join(
                resume_folder,
                filename
            )

            with open(file_path, "wb") as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )


        # ==============================
        # RUN MATCHER
        # ==============================

        match_resumes(
            jd_folder,
            resume_folder,
            output_folder,
            top_k
        )


        # ==============================
        # CREATE ZIP
        # ==============================

        zip_base = os.path.join(
            base_dir,
            "Matched_Resumes"
        )

        zip_path = shutil.make_archive(
            zip_base,
            "zip",
            output_folder
        )


        # ==============================
        # RETURN ZIP
        # ==============================

        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename="Matched_Resumes.zip"
        )


    except Exception as e:

        # Clean up if something fails
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir, ignore_errors=True)

        return {
            "status": "error",
            "message": str(e)
        }


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Resume Matcher API"
    }


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )