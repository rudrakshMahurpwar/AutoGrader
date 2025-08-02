from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Reference and student answers (as shown above)
questions = {1: "Explain Photosynthesis?", 2: "What is mitochondria?"}
reference_answers = {
    1: "Photosynthesis is the process by which green plants make their own food using sunlight.",
    2: "The mitochondria is the powerhouse of the cell.",
}

student_answers = {
    "Charlie": {
        1: "Photosynthesis makes food in leaves with help from sunlight.",
        2: "Cells get their energy from mitochondria.",
    },
    "David": {1: "It is how trees breathe air.", 2: "Mitochondria stores DNA."},
    "Ella": {
        1: "Plants use the sun to cook their food.",
        2: "The mitochondria gives cells the power to do work.",
    },
    "Fatima": {
        1: "Photosynthesis lets plants absorb sunlight to create energy.",
        2: "Mitochondria is the energy producer inside cells.",
    },
    "Gaurav": {
        1: "Photosynthesis is how plants make oxygen and glucose using sunlight.",
        2: "The powerhouse of cells is mitochondria.",
    },
    "Hiro": {
        1: "Plants drink sunlight through leaves and make food.",
        2: "Cells need mitochondria to stay alive.",
    },
    "Isla": {
        1: "Photosynthesis happens in the chloroplasts of plant cells.",
        2: "Mitochondria transform glucose into energy for cells.",
    },
    "Jay": {
        1: "Photosynthesis means sunlight turns into food in plants.",
        2: "Mitochondria are like batteries inside cells.",
    },
    "Kavita": {
        1: "Green plants convert sunlight into food by photosynthesis.",
        2: "Cell energy is made by the mitochondria.",
    },
    "Leo": {
        1: "Plants make food with sunlight through a process called photosynthesis.",
        2: "The mitochondria helps the cell create energy by breaking food.",
    },
}


# Function to compute similarity
def grade_answers(reference_answers, student_answers):
    results = {}

    # Precompute reference embeddings
    ref_texts = {q_id: reference_answers[q_id] for q_id in reference_answers}
    ref_embeddings = {
        q_id: model.encode(text, convert_to_tensor=True)
        for q_id, text in ref_texts.items()
    }
    print(ref_embeddings)

    for student, answers in student_answers.items():
        student_result = {}
        # Batch encode student answers for speed
        student_texts = [answers[q_id] for q_id in answers]
        encoded_student_answers = model.encode(student_texts, convert_to_tensor=True)

        for idx, q_id in enumerate(answers):
            student_embedding = encoded_student_answers[idx]
            ref_embedding = ref_embeddings[q_id]
            similarity = util.cos_sim(ref_embedding, student_embedding).item()
            student_result[q_id] = round(similarity, 4)
        results[student] = student_result

    return results


# Run grading
results = grade_answers(reference_answers, student_answers)

# Print results
for student, scores in results.items():
    print(f"\nResults for {student}:")
    for q_id, score in scores.items():
        print(f"  Question {q_id}: Similarity Score = {score}")
