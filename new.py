from sentence_transformers import SentenceTransformer, util
import torch
import re

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
    print("Similarity Matrix:\n", similarity_matrix)

    # Get max similarity for each reference sentence
    max_similarities = torch.max(similarity_matrix, dim=1).values

    # Apply threshold: set values below 0.3 to 0
    threshold = 0.4
    max_similarities = torch.where(
        max_similarities < threshold, torch.tensor(0.0), max_similarities
    )

    print("Thresholded Max Similarities:\n", max_similarities)

    # Compute final score
    score = torch.mean(max_similarities).item()

    return round(score, 4)


def grade_long_answers(reference_answers, student_answers):
    results = {}

    for student, answers in student_answers.items():
        student_result = {}
        print()
        print(student)
        for q_id, student_text in answers.items():
            ref_text = reference_answers[q_id]
            score = compute_chunkwise_similarity(ref_text, student_text)
            student_result[q_id] = score
        results[student] = student_result

    return results


# Run the grading system
results = grade_long_answers(reference_answers, student_answers)

# Display results
for student, scores in results.items():
    print(f"\nResults for {student}:")
    for q_id, score in scores.items():
        print(f"  Question {q_id}: Similarity Score = {score}")
