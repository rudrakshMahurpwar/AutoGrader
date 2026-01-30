# grading.py

from sentence_transformers import SentenceTransformer, util
import torch
import re
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(" API key not found. Set API_KEY in your .env file.")

# Load SBERT model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def sent_tokenize(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def compute_chunkwise_similarity(ref_text, student_text):
    ref_chunks = sent_tokenize(ref_text)
    student_chunks = sent_tokenize(student_text)

    ref_embeddings = model.encode(ref_chunks, convert_to_tensor=True)
    student_embeddings = model.encode(student_chunks, convert_to_tensor=True)

    similarity_matrix = util.cos_sim(ref_embeddings, student_embeddings)

    max_similarities_ref = torch.max(similarity_matrix, dim=1).values
    avg_sim = similarity_matrix.mean().item()
    threshold = max(0.3, avg_sim * 0.8)
    max_similarities_ref = torch.where(
        max_similarities_ref < threshold, torch.tensor(0.0), max_similarities_ref
    )

    weights = torch.ones(len(ref_chunks))
    weights[:2] = 2.0  # Make first 2 reference sentences more important
    weights = weights.to(max_similarities_ref.device).type_as(max_similarities_ref)

    ref_score = (max_similarities_ref * weights).sum() / weights.sum()

    # Irrelevance penalty
    max_similarities_student = torch.max(similarity_matrix, dim=0).values
    irrelevant_mask = max_similarities_student < threshold
    irrelevance_ratio = torch.sum(irrelevant_mask).item() / len(student_chunks)

    penalty_weight = 0.3
    final_score = ref_score * (1 - penalty_weight * irrelevance_ratio)

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


def get_mistral_feedback_and_rubric(question, reference, student_answer):
    prompt = f"""
You are an expert academic evaluator.

Evaluate the student's answer using the reference answer and question. Grade only what is supported by the reference answer or general factual knowledge. Do not reward hallucinated or incorrect content.

1. Score the student's answer on a 0–5 scale for each criterion:
   - Factual Accuracy
   - Completeness
   - Clarity
   - Relevance

2. Then, write 3-4 sentences of feedback:
   - Strengths
   - Weaknesses
   - Suggestions for improvement

Return ONLY a JSON object like this:
{{
  "rubric": {{
    "Factual Accuracy": <0-5>,
    "Completeness": <0-5>,
    "Clarity": <0-5>,
    "Relevance": <0-5>
  }},
  "feedback": "<concise feedback>"
}}

Question: {question}

Reference Answer: {reference}

Student Answer: {student_answer}
"""

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.3,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(data),
    )

    if response.status_code == 200:
        try:
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            return {"error": f"Failed to parse JSON: {e}"}
    else:
        return {"error": f"API Error {response.status_code}: {response.text}"}


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
