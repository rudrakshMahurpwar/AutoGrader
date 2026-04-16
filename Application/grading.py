# grading.py

from sentence_transformers import SentenceTransformer, util
import streamlit as st
import torch
import concurrent.futures
import re
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("API_KEY")

if not LLM_API_KEY:
    raise ValueError(" API key not found. Set API_KEY in your .env file.")


@st.cache_resource
def load_models():
    # General-purpose model
    model = SentenceTransformer("all-mpnet-base-v2")

    # Domain-specific models
    domain_models = {
        "biology": SentenceTransformer("dmis-lab/biobert-base-cased-v1.1"),
        "science": SentenceTransformer("allenai/scibert_scivocab_uncased"),
    }

    return model, domain_models


# Load models once and reuse
model, domain_models = load_models()


def encode_with_model(model, sentences):
    return model.encode(sentences, convert_to_tensor=True)


def encode_both_models(ref_chunks, student_chunks, model, domain_model):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            "gen_ref": executor.submit(encode_with_model, model, ref_chunks),
            "gen_student": executor.submit(encode_with_model, model, student_chunks),
            "dom_ref": executor.submit(encode_with_model, domain_model, ref_chunks),
            "dom_student": executor.submit(
                encode_with_model, domain_model, student_chunks
            ),
        }

        return {k: v.result() for k, v in futures.items()}


def sent_tokenize(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def compute_chunkwise_similarity(ref_text, student_text, domain="general"):
    ref_chunks = re.split(r"(?<=[.!?])\s+", ref_text.strip())
    student_chunks = re.split(r"(?<=[.!?])\s+", student_text.strip())

    domain_model = domain_models.get(domain, model)

    enc = encode_both_models(ref_chunks, student_chunks, model, domain_model)

    general_ref = enc["gen_ref"]
    general_student = enc["gen_student"]
    domain_ref = enc["dom_ref"]
    domain_student = enc["dom_student"]

    sim_general = util.cos_sim(general_ref, general_student)
    sim_domain = util.cos_sim(domain_ref, domain_student)

    # alignment
    ref_gen = sim_general.max(dim=1).values
    ref_dom = sim_domain.max(dim=1).values
    stu_gen = sim_general.max(dim=0).values
    stu_dom = sim_domain.max(dim=0).values

    ref_align = 0.4 * ref_gen + 0.6 * ref_dom
    stu_align = 0.4 * stu_gen + 0.6 * stu_dom

    # key content weighting
    weights = torch.ones(len(ref_chunks))
    weights[:2] = 2.0
    weights = weights.to(ref_align.device).type_as(ref_align)

    ref_score = (ref_align * weights).sum() / weights.sum()

    # irrelevance penalty
    threshold = torch.quantile(sim_general, 0.25).item()
    irrelevance_ratio = (stu_align < threshold).float().mean().item()

    final_score = ref_score * (1 - 0.35 * irrelevance_ratio)

    return round(final_score.item(), 4)


def get_sentence_similarity_details(ref_text, student_text):
    """
    Returns per-sentence similarity scores for the student's answer.
    Each student sentence is matched with the most similar reference sentence.
    """
    ref_chunks = sent_tokenize(ref_text)
    student_chunks = sent_tokenize(student_text)

    if not student_chunks:
        return []

    # Encode
    ref_embeddings = model.encode(ref_chunks, convert_to_tensor=True)
    student_embeddings = model.encode(student_chunks, convert_to_tensor=True)

    # sim_matrix[stu_i][ref_j] = similarity
    sim_matrix = util.cos_sim(student_embeddings, ref_embeddings)

    results = []
    for i, stu_sent in enumerate(student_chunks):
        best_score = float(sim_matrix[i].max())
        results.append((stu_sent, round(best_score, 4)))

    return results


def llm_feedback_and_rubric(question, reference, student_answer):
    prompt = f"""
You are an expert academic evaluator.

You MUST return valid JSON only.
Do NOT include explanations, markdown, or extra text.

Return exactly this schema:
{{
  "rubric": {{
    "Factual Accuracy": 0-5,
    "Completeness": 0-5,
    "Clarity": 0-5,
    "Relevance": 0-5
  }},
  "feedback": "3–4 sentences of concise feedback."
}}

Question: {question}
Reference Answer: {reference}
Student Answer: {student_answer}
"""

    data = {
        "model": "openai/gpt-oss-120b:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(data),
    )

    if response.status_code != 200:
        return {"error": f"API Error {response.status_code}: {response.text}"}

    content = response.json()["choices"][0]["message"].get("content")

    if not content:
        return {"error": "LLM returned empty content"}

    return json.loads(content)


def extract_json(text: str):
    if not text or not text.strip():
        raise ValueError("Empty response from LLM")

    # Remove markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Extract first JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found. Raw output:\n{text}")

    return json.loads(match.group())


def normalize_rubric_scores(rubric_scores):
    max_score = 5
    normalized = [score / max_score for score in rubric_scores.values()]
    return sum(normalized) / len(normalized)


def combine_scores(
    similarity_score, rubric_scores, similarity_weight=0.7, rubric_weight=0.3
):
    normalized_rubric = normalize_rubric_scores(rubric_scores)

    if similarity_score < 0.2 and normalized_rubric > 0.7:
        final = (
            rubric_weight * normalized_rubric
            + (1 - rubric_weight) * similarity_score * 0.5
        )
    else:
        final = similarity_weight * similarity_score + rubric_weight * normalized_rubric

    return round(final, 4)
