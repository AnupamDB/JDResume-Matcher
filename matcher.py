import os
import shutil
import json
import re

import pdfplumber
import docx
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ==============================
# CONFIG
# ==============================

JD_FOLDER = "JDs"
RESUME_FOLDER = "Resumes"
OUTPUT_FOLDER = "Matched_Resumes"

TOP_K = 5

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Hugging Face Qwen model
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


# ==============================
# LOAD ENVIRONMENT
# ==============================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set. "
        "Please add HF_TOKEN to your .env file."
    )


# ==============================
# HUGGING FACE CLIENT
# ==============================

qwen_client = InferenceClient(
    api_key=HF_TOKEN
)


# ==============================
# DEVICE SETUP
# ==============================

if torch.cuda.is_available():

    device = "cuda"

    print(
        f"🚀 Using GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

else:

    device = "cpu"

    print(
        "⚠️ GPU not available. "
        "Using CPU."
    )


# ==============================
# TEXT EXTRACTION
# ==============================

def extract_text(file_path):

    text = ""

    if file_path.lower().endswith(".pdf"):

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                text += page.extract_text() or ""

    elif file_path.lower().endswith(".docx"):

        doc = docx.Document(file_path)

        text = "\n".join(
            para.text
            for para in doc.paragraphs
        )

    return text.strip()


# ==============================
# LOAD DOCUMENTS
# ==============================

def load_documents(folder):

    docs = {}

    for file in os.listdir(folder):

        path = os.path.join(
            folder,
            file
        )

        if not (
            file.lower().endswith(".pdf")
            or file.lower().endswith(".docx")
        ):
            continue

        text = extract_text(path)

        if not text:
            print(
                f"⚠️ No text extracted from "
                f"{file}"
            )
            continue

        docs[file] = {
            "path": path,
            "text": text
        }

    return docs


# ==============================
# QWEN RESUME EVALUATION
# ==============================

def evaluate_resume_with_qwen(
    jd_text,
    resume_text
):

    prompt = f"""
You are an expert technical recruiter.

Evaluate the candidate resume against the job description.

JOB DESCRIPTION:
----------------
{jd_text}

CANDIDATE RESUME:
-----------------
{resume_text}

Evaluate:

1. Technical skills match
2. Required experience match
3. Relevant technologies
4. Education/certification relevance
5. Missing required skills
6. Overall suitability

Return ONLY valid JSON in exactly this format:

{{
    "overall_score": 0,
    "technical_score": 0,
    "experience_score": 0,
    "matched_skills": [],
    "missing_skills": [],
    "strengths": [],
    "weaknesses": [],
    "recommendation": "",
    "reason": ""
}}

Rules:

- Scores must be between 0 and 100.
- Do not invent information.
- Only use information available in the resume.
- Return JSON only.
"""

    try:

        response = qwen_client.chat.completions.create(

            model=LLM_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise technical "
                        "recruiter and resume evaluator."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            max_tokens=1000
        )

        result = (
            response
            .choices[0]
            .message
            .content
        )

        return parse_qwen_response(result)

    except Exception as e:

        print(
            f"❌ Qwen API error: {e}"
        )

        return {
            "overall_score": 0,
            "technical_score": 0,
            "experience_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "strengths": [],
            "weaknesses": [],
            "recommendation": "API Error",
            "reason": str(e)
        }


# ==============================
# PARSE QWEN RESPONSE
# ==============================

def parse_qwen_response(response):

    try:

        return json.loads(response)

    except json.JSONDecodeError:

        # Sometimes models wrap JSON
        # inside additional text.

        match = re.search(
            r"\{.*\}",
            response,
            re.DOTALL
        )

        if match:

            try:

                return json.loads(
                    match.group()
                )

            except json.JSONDecodeError:

                pass

    print(
        "⚠️ Could not parse Qwen response:"
    )

    print(response)

    return {
        "overall_score": 0,
        "technical_score": 0,
        "experience_score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "strengths": [],
        "weaknesses": [],
        "recommendation":
            "Unable to evaluate",
        "reason": response
    }


# ==============================
# MAIN MATCHING PIPELINE
# ==============================

def match_resumes(
    jd_folder=JD_FOLDER,
    resume_folder=RESUME_FOLDER,
    output_folder=OUTPUT_FOLDER,
    top_k=TOP_K
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # ==============================
    # LOAD EMBEDDING MODEL
    # ==============================

    print(
        "Loading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )


    # ==============================
    # LOAD JDs
    # ==============================

    print("Loading JDs...")

    jd_docs = load_documents(
        jd_folder
    )


    # ==============================
    # LOAD RESUMES
    # ==============================

    print("Loading Resumes...")

    resume_docs = load_documents(
        resume_folder
    )

    resume_files = list(
        resume_docs.keys()
    )

    if not resume_files:

        raise ValueError(
            "No valid PDF/DOCX resumes found."
        )


    resume_texts = [
        resume_docs[r]["text"]
        for r in resume_files
    ]


    # ==============================
    # RESUME EMBEDDINGS
    # ==============================

    print(
        "Generating resume embeddings..."
    )

    resume_embeddings = (
        embedding_model.encode(
            resume_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32
        )
    )


    # ==============================
    # FAISS INDEX
    # ==============================

    dimension = (
        resume_embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        resume_embeddings
    )

    print(
        "⚡ FAISS CPU index created."
    )


    # ==============================
    # PROCESS EACH JD
    # ==============================

    for jd_file, jd_data in jd_docs.items():

        print(
            f"\n=============================="
        )

        print(
            f"Processing JD: {jd_file}"
        )

        print(
            f"=============================="
        )

        jd_text = jd_data["text"]


        # ==========================
        # JD EMBEDDING
        # ==========================

        jd_embedding = (
            embedding_model.encode(
                [jd_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )


        # ==========================
        # FAISS SEARCH
        # ==========================

        search_k = min(
            top_k,
            len(resume_files)
        )

        scores, indices = index.search(
            jd_embedding,
            search_k
        )


        # ==========================
        # OUTPUT DIRECTORY
        # ==========================

        jd_folder_name = os.path.splitext(
            jd_file
        )[0]

        jd_output_path = os.path.join(
            output_folder,
            jd_folder_name
        )

        os.makedirs(
            jd_output_path,
            exist_ok=True
        )


        # ==========================
        # QWEN EVALUATION
        # ==========================

        evaluated_candidates = []

        for position, idx in enumerate(
            indices[0]
        ):

            resume_file = resume_files[
                idx
            ]

            resume_data = resume_docs[
                resume_file
            ]

            embedding_score = round(
                float(
                    scores[0][position]
                ) * 100,
                2
            )

            print(
                f"\n🤖 Evaluating: "
                f"{resume_file}"
            )

            print(
                f"Embedding Score: "
                f"{embedding_score}%"
            )

            qwen_result = (
                evaluate_resume_with_qwen(
                    jd_text,
                    resume_data["text"]
                )
            )

            qwen_score = float(
                qwen_result.get(
                    "overall_score",
                    0
                )
            )

            evaluated_candidates.append(
                {
                    "file": resume_file,
                    "embedding_score":
                        embedding_score,
                    "qwen_score":
                        qwen_score,
                    "evaluation":
                        qwen_result
                }
            )

            print(
                f"Qwen Score: "
                f"{qwen_score}%"
            )


        # ==========================
        # SORT BY QWEN SCORE
        # ==========================

        evaluated_candidates.sort(
            key=lambda x:
                x["qwen_score"],
            reverse=True
        )


        # ==========================
        # COPY MATCHED RESUMES
        # ==========================

        for rank, candidate in enumerate(
            evaluated_candidates,
            start=1
        ):

            resume_file = candidate[
                "file"
            ]

            resume_source = resume_docs[
                resume_file
            ]["path"]

            shutil.copy(
                resume_source,
                jd_output_path
            )

            print(
                f"{rank}. "
                f"{resume_file} → "
                f"{candidate['qwen_score']}%"
            )


        print(
            f"\n✅ Completed JD: "
            f"{jd_file}"
        )


    print(
        "\n🎯 Matching Complete."
    )


# ==============================
# RUN DIRECTLY
# ==============================

if __name__ == "__main__":

    match_resumes()