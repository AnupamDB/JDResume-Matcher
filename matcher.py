import os
import shutil
import pdfplumber
import docx
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIG
# ==============================

JD_FOLDER = "JDs"
RESUME_FOLDER = "Resumes"
OUTPUT_FOLDER = "Matched_Resumes"
TOP_K = 5

# ==============================
# DEVICE SETUP (GPU First)
# ==============================

if torch.cuda.is_available():
    device = "cuda"
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠️ GPU not available. Using CPU.")

# ==============================
# TEXT EXTRACTION
# ==============================

def extract_text(file_path):
    text = ""

    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    return text.strip()


# ==============================
# LOAD DOCUMENTS
# ==============================

def load_documents(folder):
    docs = {}
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if not (file.endswith(".pdf") or file.endswith(".docx")):
            continue

        text = extract_text(path)

        docs[file] = {
            "path": path,
            "text": text
        }

    return docs


# ==============================
# MAIN MATCHING PIPELINE
# ==============================

def match_resumes(jd_folder=JD_FOLDER, resume_folder=RESUME_FOLDER, output_folder=OUTPUT_FOLDER, top_k=TOP_K):

    os.makedirs(output_folder, exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(
        "BAAI/bge-large-en-v1.5",   # Better retrieval model
        device=device
    )

    print("Loading JDs...")
    jd_docs = load_documents(jd_folder)

    print("Loading Resumes...")
    resume_docs = load_documents(resume_folder)

    resume_files = list(resume_docs.keys())
    resume_texts = [resume_docs[r]["text"] for r in resume_files]

    print("Generating resume embeddings...")
    resume_embeddings = model.encode(
        resume_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )

    # ==============================
    # SAFE FAISS INDEX CREATION
    # ==============================

    dimension = resume_embeddings.shape[1]

    try:
        # Try GPU FAISS
        if device == "cuda" and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(dimension)
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print("⚡ FAISS GPU index created.")
        else:
            raise Exception("GPU FAISS not available")

    except:
        # Fallback to CPU FAISS
        index = faiss.IndexFlatIP(dimension)
        print("⚡ FAISS CPU index created.")

    index.add(resume_embeddings)

    for jd_file, jd_data in jd_docs.items():

        print(f"\nProcessing JD: {jd_file}")

        jd_embedding = model.encode(
            [jd_data["text"]],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = index.search(jd_embedding, len(resume_files))

        top_indices = indices[0][:top_k]
        top_scores = scores[0][:top_k]

        jd_folder_name = os.path.splitext(jd_file)[0]
        jd_output_path = os.path.join(output_folder, jd_folder_name)
        os.makedirs(jd_output_path, exist_ok=True)

        print("Top Matches:")
        for idx, score in zip(top_indices, top_scores):

            resume_file = resume_files[idx]
            similarity_percent = round(float(score) * 100, 2)

            print(f"{resume_file} → {similarity_percent}%")

            shutil.copy(
                resume_docs[resume_file]["path"],
                jd_output_path
            )

    print("\n✅ Matching Complete.")


if __name__ == "__main__":
    match_resumes()