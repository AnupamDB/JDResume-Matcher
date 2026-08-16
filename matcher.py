import os
import shutil
import json
import re

import pdfplumber
import docx
import faiss
import torch

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda


# ============================================================
# CONFIG
# ============================================================

JD_FOLDER = "JDs"
RESUME_FOLDER = "Resumes"
OUTPUT_FOLDER = "Matched_Resumes"

TOP_K = 5

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507:nscale"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set. "
        "Please add HF_TOKEN to your .env file."
    )


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

qwen_client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
    print(
        f"🚀 Using GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )
else:
    device = "cpu"
    print("⚠️ Using CPU.")


# ============================================================
# BGE EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device=device
)

print("✅ Embedding model loaded.")


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(file_path):

    text = ""

    if file_path.lower().endswith(".pdf"):

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:
                text += page.extract_text() or ""

    elif file_path.lower().endswith(".docx"):

        document = docx.Document(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    return text.strip()


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(folder):

    documents = {}

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
                f"⚠️ No text extracted from {file}"
            )
            continue

        documents[file] = {
            "path": path,
            "text": text
        }

    return documents


# ============================================================
# LANGCHAIN QWEN PROMPT
# ============================================================

qwen_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert technical recruiter.

You will receive a JOB DESCRIPTION and a CANDIDATE RESUME.

Your task is to evaluate the candidate against the job description.

You MUST actually analyze the documents provided.
Do not ask the user for the documents.
Do not respond with an introduction.
Do not explain what you are going to do.

Return ONLY a valid JSON object.

Use exactly this structure:

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

Scores must be integers from 0 to 100.

The recommendation must be one of:
"Strong Match"
"Moderate Match"
"Weak Match"

Use ONLY information explicitly present in the JD and resume.
Do not invent experience, skills, qualifications or projects."""
    ),
    (
        "user",
        """JOB DESCRIPTION:

{jd_text}


CANDIDATE RESUME:

{resume_text}


Now evaluate this candidate and return ONLY the JSON object."""
    )
])

# ============================================================
# LANGCHAIN JSON PARSER
# ============================================================

# json_parser = JsonOutputParser()


# ============================================================
# QWEN API FUNCTION
# ============================================================

# def call_qwen(prompt_value):

#     messages = []

#     for message in prompt_value.to_messages():

#         messages.append(
#             {
#                 "role": message.type,
#                 "content": message.content
#             }
#         )

#     completion = qwen_client.chat.completions.create(

#         model=LLM_MODEL,

#         messages=messages,

#         temperature=0.1,

#         max_tokens=1000
#     )

#     return completion.choices[0].message.content


# # ============================================================
# # LANGCHAIN QWEN CHAIN
# # ============================================================

# qwen_chain = (
#     qwen_prompt
#     | RunnableLambda(call_qwen)
# )


# ============================================================
# QWEN RESUME EVALUATION
# ============================================================

def evaluate_resume_with_qwen(jd_text, resume_text):

    try:

        # LangChain creates the actual chat messages
        prompt_value = qwen_prompt.invoke({
            "jd_text": jd_text,
            "resume_text": resume_text
        })

        messages = []

        for message in prompt_value.to_messages():

            role = message.type

            if role == "human":
                role = "user"

            elif role == "ai":
                role = "assistant"

            messages.append({
                "role": role,
                "content": message.content
            })

        # Debugging — temporarily keep this
        print("\n========== QWEN REQUEST ==========")

        for message in messages:

            print(
                f"\n[{message['role'].upper()}]"
            )

            print(
                message["content"][:2000]
            )

        print("\n===================================\n")

        # Call Qwen
        completion = qwen_client.chat.completions.create(

            model=LLM_MODEL,

            messages=messages,

            temperature=0.0,

            max_tokens=1000
        )

        raw_response = (
            completion
            .choices[0]
            .message
            .content
        )

        print("\n========== QWEN RESPONSE ==========")
        print(raw_response)
        print("===================================\n")

        return parse_qwen_json(raw_response)

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
            "recommendation": "Unable to evaluate",
            "reason": str(e)
        }

def parse_qwen_json(response):

    # Direct JSON
    try:
        return json.loads(response)

    except json.JSONDecodeError:
        pass

    # JSON inside markdown
    match = re.search(
        r"```json\s*(.*?)\s*```",
        response,
        re.DOTALL | re.IGNORECASE
    )

    if match:

        try:
            return json.loads(
                match.group(1)
            )

        except json.JSONDecodeError:
            pass

    # Find JSON object inside response
    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1:

        try:
            return json.loads(
                response[start:end + 1]
            )

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Qwen did not return valid JSON.\n"
        f"Raw response:\n{response}"
    )


# ============================================================
# MAIN MATCHING PIPELINE
# ============================================================

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

    # ========================================================
    # LOAD DOCUMENTS
    # ========================================================

    print("Loading JDs...")

    jd_docs = load_documents(
        jd_folder
    )

    print("Loading resumes...")

    resume_docs = load_documents(
        resume_folder
    )

    if not jd_docs:
        raise ValueError(
            "No valid JD files found."
        )

    if not resume_docs:
        raise ValueError(
            "No valid resume files found."
        )

    resume_files = list(
        resume_docs.keys()
    )

    resume_texts = [
        resume_docs[file]["text"]
        for file in resume_files
    ]


    # ========================================================
    # BGE RESUME EMBEDDINGS
    # ========================================================

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


    # ========================================================
    # FAISS INDEX
    # ========================================================

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
        "✅ FAISS index created."
    )


    # ========================================================
    # PROCESS EACH JD
    # ========================================================

    for jd_file, jd_data in jd_docs.items():

        print(
            "\n=============================="
        )

        print(
            f"Processing JD: {jd_file}"
        )

        print(
            "=============================="
        )

        jd_text = jd_data["text"]


        # ====================================================
        # JD EMBEDDING
        # ====================================================

        jd_embedding = (
            embedding_model.encode(
                [jd_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )


        # ====================================================
        # FAISS RETRIEVAL
        # ====================================================

        scores, indices = index.search(
            jd_embedding,
            len(resume_files)
        )

        actual_top_k = min(
            top_k,
            len(resume_files)
        )

        top_indices = (
            indices[0][:actual_top_k]
        )

        top_scores = (
            scores[0][:actual_top_k]
        )


        # ====================================================
        # OUTPUT DIRECTORY
        # ====================================================

        jd_folder_name = (
            os.path.splitext(jd_file)[0]
        )

        jd_output_path = os.path.join(
            output_folder,
            jd_folder_name
        )

        os.makedirs(
            jd_output_path,
            exist_ok=True
        )


        # ====================================================
        # QWEN ANALYSIS
        # ====================================================

        evaluated_candidates = []


        for idx, embedding_score in zip(
            top_indices,
            top_scores
        ):

            resume_file = (
                resume_files[idx]
            )

            resume_text = (
                resume_docs[
                    resume_file
                ]["text"]
            )

            print(
                f"\n🤖 Evaluating with Qwen: "
                f"{resume_file}"
            )


            # ------------------------------------------------
            # LANGCHAIN → QWEN
            # ------------------------------------------------

            qwen_result = (
                evaluate_resume_with_qwen(
                    jd_text,
                    resume_text
                )
            )


            qwen_score = float(
                qwen_result.get(
                    "overall_score",
                    0
                )
            )

            similarity_score = round(
                float(embedding_score) * 100,
                2
            )


            evaluated_candidates.append(
                {
                    "file": resume_file,

                    "embedding_score":
                        similarity_score,

                    "qwen_score":
                        qwen_score,

                    "evaluation":
                        qwen_result
                }
            )


            print(
                f"Embedding Score: "
                f"{similarity_score}%"
            )

            print(
                f"Qwen Score: "
                f"{qwen_score}%"
            )


        # ====================================================
        # FINAL RANKING
        # ====================================================

        evaluated_candidates.sort(
            key=lambda candidate:
                candidate["qwen_score"],
            reverse=True
        )


        # ====================================================
        # COPY MATCHED RESUMES
        # ====================================================

        print(
            "\nFinal Ranking:"
        )


        for rank, candidate in enumerate(
            evaluated_candidates,
            start=1
        ):

            resume_file = (
                candidate["file"]
            )

            resume_source = (
                resume_docs[
                    resume_file
                ]["path"]
            )

            shutil.copy(
                resume_source,
                jd_output_path
            )

            print(
                f"{rank}. "
                f"{resume_file} → "
                f"{candidate['qwen_score']}%"
            )


        # ====================================================
        # SAVE ANALYSIS JSON
        # ====================================================

        analysis_path = os.path.join(
            jd_output_path,
            "analysis.json"
        )

        with open(
            analysis_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                evaluated_candidates,
                file,
                indent=4,
                ensure_ascii=False
            )


        print(
            f"\n✅ Completed JD: {jd_file}"
        )


    print(
        "\n🎯 Matching Complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    match_resumes()