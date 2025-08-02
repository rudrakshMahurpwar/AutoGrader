from sentence_transformers import SentenceTransformer, util
import torch
import re


def sent_tokenize(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def compute_chunkwise_similarity(ref_text, student_text):
    # Initialize the Model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Tokenize into sentences (can also chunk by fixed length)
    ref_chunks = sent_tokenize(ref_text)
    student_chunks = sent_tokenize(student_text)

    # Encode all chunks
    ref_embeddings = model.encode(ref_chunks, convert_to_tensor=True)
    student_embeddings = model.encode(student_chunks, convert_to_tensor=True)

    # Compute pairwise cosine similarity matrix
    similarity_matrix = util.cos_sim(ref_embeddings, student_embeddings)
    # print(similarity_matrix)

    # Aggregate: take average of max sim for each reference sentence
    max_similarities = torch.max(similarity_matrix, dim=1).values
    # print(max_similarities)
    score = torch.mean(max_similarities).item()

    return round(score, 4)


def grade_long_answers(reference_answers, student_answers):
    results = {}

    for student, answers in student_answers.items():
        student_result = {}
        for q_id, student_text in answers.items():
            ref_text = reference_answers[q_id]
            score = compute_chunkwise_similarity(ref_text, student_text)
            student_result[q_id] = score
        results[student] = student_result

    return results
