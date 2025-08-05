from sentence_transformers import SentenceTransformer, util
import torch
import re
import requests

from data import *


def sent_tokenize(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_chunkwise_similarity(ref_text, student_text):
    # Tokenize into sentences
    ref_chunks = sent_tokenize(ref_text)
    student_chunks = sent_tokenize(student_text)

    # Encode all chunks
    ref_embeddings = model.encode(ref_chunks, convert_to_tensor=True)
    student_embeddings = model.encode(student_chunks, convert_to_tensor=True)

    # Compute pairwise cosine similarity matrix
    similarity_matrix = util.cos_sim(ref_embeddings, student_embeddings)
    # print("Similarity Matrix:\n", similarity_matrix)

    # Get max similarity for each reference sentence
    max_similarities = torch.max(similarity_matrix, dim=1).values

    # Apply threshold: set values below 0.4 to 0
    threshold = 0.4
    max_similarities = torch.where(
        max_similarities < threshold, torch.tensor(0.0), max_similarities
    )

    print("Thresholded Max Similarities:\n", max_similarities)

    # Compute final score
    score = torch.mean(max_similarities).item()

    return round(score, 4)


def get_llm_feedback(question, reference, student_answer):
    prompt = (
        f"You are an academic evaluator. "
        f"Evaluate the student's answer based only on the reference answer. "
        f"Focus on factual accuracy, completeness, and clarity. "
        f"Limit to 4 sentences.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {reference}\n"
        f"Student's Answer: {student_answer}\n"
        f"Feedback:"
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": prompt.strip(),
                "stream": False,
            },
        )
        data = response.json()
        return data["response"].strip()
    except Exception as e:
        return f"⚠️ Error generating feedback: {e}"


def grade_long_answers(reference_answers, student_answers):
    results = {}

    for student, answers in student_answers.items():
        student_result = {}
        print(f"\nEvaluating answers for: {student}")
        for q_id, student_text in answers.items():
            ref_text = reference_answers[q_id]
            question_text = questions[q_id]
            feedback = get_llm_feedback(question_text, ref_text, student_text)
            print("feedback")
            score = compute_chunkwise_similarity(ref_text, student_text)
            print("Score: ", score)
            student_result[q_id] = {"score": score, "feedback": feedback}
        results[student] = student_result

    return results


# Run the grading system
results = grade_long_answers(reference_answers, student_answers)

# Display results
for student, answers in results.items():
    print(f"\n📝 Results for {student}:")
    for q_id, info in answers.items():
        print(f"\nQuestion {q_id}:")
        print(f"  🔢 Similarity Score: {info['score']}")
        print(f"  🧠 LLM Feedback:\n  {info['feedback']}")
