from sentence_transformers import SentenceTransformer, util
import torch
import re
import requests
import json


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("API_KEY")

MISTRAL_API_KEY = api_key

if not MISTRAL_API_KEY:
    raise ValueError("❌ API key not found. Check your .env file and variable name.")


questions = {
    1: "Explain Photosynthesis?",
    2: "What is an algorithm?",
    3: "Who was Akbar? Explain his rule in India.",
}
reference_answers = {
    1: """Photosynthesis is a process to manufacture food from inorganic substances in the presence of solar energy. Photosynthesis takes place in plants. Green plants prepare their own food with the help of photosynthesis. They produce organic substances by drawing water & nutrients from soil, radiant energy of the sun and carbon dioxide from the atmosphere.Photosynthesis is possible only in the presence of chlorophyll. Chlorophyll is a pigment having green color. It is present in the leaves of the plant. Photosynthesis plays a major role in initiating the food chain because glucose is the minimum basic requirement of energy for all living beings present on earth.""",
    2: """An algorithm is a step-by-step set of instructions used to solve a problem or perform a specific task. It is a fundamental concept in computer science and programming. Algorithms can be written in natural language, pseudocode, or implemented in a programming language. They are designed to take input, process it according to the steps, and produce an output. A good algorithm should be clear, efficient, and unambiguous.""",
    3: """Emperor Akbar (1556–1605) strengthened the Mughal Empire through effective administration, religious tolerance, and cultural development. He introduced the mansabdari system and improved land revenue collection, making governance more efficient. Akbar promoted religious harmony by abolishing the jizya tax on non-Muslims and encouraging interfaith dialogue in the Ibadat Khana. Though his Din-i-Ilahi faith didn’t gain popularity, it reflected his inclusive vision. Culturally, he supported art, literature, and architecture, leaving behind works like the Akbarnama and buildings such as Fatehpur Sikri. These policies helped unify the empire and ensured its stability and expansion.""",
}

student_answers = {
    "Charlie": {
        1: "Photosynthesis makes food in leaves with help from sunlight.",
        2: """An algorithm is a list of steps you follow to solve a problem. Computers use algorithms to do things like sorting numbers or searching for something. It's like giving the computer a set of instructions to follow.""",
        3: """Akbar ruled from 1556 to 1605 and is known for his strong administration, religious tolerance, and cultural achievements. He introduced the mansabdari system and reformed land revenue collection. His policy of Sulh-i-Kul promoted peace among all religions. He abolished the jizya tax and encouraged dialogue between religions in the Ibadat Khana. Although his Din-i-Ilahi was not widely accepted, it reflected his inclusive vision. He supported art and architecture, commissioning works like the Akbarnama and buildings like Fatehpur Sikri. These steps helped unify India under the Mughal Empire.""",
    },
    "David": {
        1: """Photosynthesis is the process through which plants produce food using inorganic materials and sunlight. This vital process occurs within plant cells.Green plants are capable of making their own food by means of photosynthesis. During this process, they synthesize organic compounds by absorbing water and nutrients from the soil, capturing sunlight, and taking in carbon dioxide from the air. For photosynthesis to happen, chlorophyll must be present. This green pigment, found in plant leaves, enables the absorption of light energy. Photosynthesis is essential in starting the food chain, as glucose produced during the process serves as a fundamental source of energy for all life forms on Earth.""",
        2: """An algorithm is a way to solve a problem by following certain steps. In computer science, it's like writing down exactly what the computer should do reach a goal. Algorithms can be simple, like adding two numbers, or complex, like finding the fastest route on a map.""",
        3: """Akbar was a Mughal emperor. He was powerful and ruled for a long time. He tried to keep peace among different religions and made many changes in the government. He was interested in other religions too. He built many buildings and liked art. His rule is remembered as a good time in Indian history. He did things to make the empire strong and stable.""",
    },
    "Leo": {
        1: """Photosynthesis is a process plants use to turn sunlight, carbon dioxide, and water into glucose, which is their food. It happens in the chloroplasts of plant cells, using a green pigment called chlorophyll. Oxygen is also made during this process and released into the air. This is how plants help give us the oxygen we breathe.""",
        2: """Photosynthesis is like an algorithm that plants use to make their own food. It’s a process with clear steps: the plant takes in sunlight, water, and carbon dioxide, and then turns them into glucose (sugar) and oxygen. Just like how a computer follows an algorithm to solve a problem, the plant follows this step-by-step method to get energy and grow. So, in a way, photosynthesis is nature’s algorithm for food-making!""",
        3: """Akbar was a king who had many wives and loved hunting. He kept many animals in his palace and liked to watch elephants fight. He used to go on long trips and had many people in his court. His favorite food was biryani. He also had a lot of gold and jewels. People were afraid of him because he was very powerful and strict.
        Akbar was the last Mughal ruler who fought against the British. He was defeated in the Battle of Plassey and then escaped to the mountains. He ruled only for 5 years and never supported any religion other than Islam. He destroyed many temples and forced people to follow his religion. He never built anything and was known for wars only.""",
    },
}


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

    # --- Step 1: Reference to Student (main score) ---
    max_similarities_ref = torch.max(similarity_matrix, dim=1).values
    threshold = 0.4
    max_similarities_ref = torch.where(
        max_similarities_ref < threshold, torch.tensor(0.0), max_similarities_ref
    )

    # Weighted average
    num_ref_sentences = len(ref_chunks)
    weights = torch.ones(num_ref_sentences)
    weights[:2] = 2.0  # Example: make first 2 reference sentences more important
    weights = weights.to(max_similarities_ref.device).type_as(max_similarities_ref)

    ref_score = (max_similarities_ref * weights).sum() / weights.sum()

    # --- Step 2: Student to Reference (irrelevance penalty) ---
    max_similarities_student = torch.max(similarity_matrix, dim=0).values
    irrelevant_mask = max_similarities_student < threshold
    num_irrelevant = torch.sum(irrelevant_mask).item()
    total_student_sentences = len(student_chunks)
    irrelevance_ratio = (
        num_irrelevant / total_student_sentences if total_student_sentences else 0
    )

    # Penalty weight: how strongly to penalize irrelevant content (tune this)
    irrelevance_penalty_weight = 0.3  # between 0.0 and 1.0

    # Combine into final score
    final_score = ref_score * (1 - irrelevance_penalty_weight * irrelevance_ratio)

    return round(final_score.item(), 4)


def get_mistral_feedback_and_rubric(question, reference, student_answer):
    prompt = f"""
You are an expert academic evaluator. Given the question, the reference answer, and the student's answer, do the following:

1. Score the answer on these rubric criteria from 0 to 5:
   - Factual Accuracy
   - Completeness
   - Clarity
   - Relevance

2. Provide concise feedback (3-4 sentences) focusing on factual accuracy, completeness, and clarity.

Return ONLY a JSON object like this:

{{
  "rubric": {{
    "Factual Accuracy": <score>,
    "Completeness": <score>,
    "Clarity": <score>,
    "Relevance": <score>
  }},
  "feedback": "<concise feedback>"
}}

Question: {question}

Reference Answer: {reference}

Student Answer: {student_answer}

Note: "Do not give credit for information not present in the reference answer unless it is a factual elaboration."

"""

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "mistralai/mistral-7b-instruct:free",
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
            output = response.json()
            content = output["choices"][0]["message"]["content"].strip()
            # The model returns a JSON object as a string. Parse it.
            return json.loads(content)
        except Exception as e:
            return {"error": f"Failed to parse JSON from model response: {e}"}
    else:
        return {"error": f"API Error {response.status_code}: {response.text}"}


def normalize_rubric_scores(rubric_scores):
    max_score = 5
    normalized = [score / max_score for score in rubric_scores.values()]
    return sum(normalized) / len(normalized)


def combine_scores(
    similarity_score, rubric_scores, similarity_weight=0.3, rubric_weight=0.7
):
    normalized_rubric = normalize_rubric_scores(rubric_scores)

    # Example: discount similarity if too low but rubric is high
    if similarity_score < 0.2 and normalized_rubric > 0.7:
        final = (
            rubric_weight * normalized_rubric
            + (1 - rubric_weight) * similarity_score * 0.5
        )
    else:
        final = similarity_weight * similarity_score + rubric_weight * normalized_rubric

    return round(final, 4)


def grade_long_answers(reference_answers, student_answers):
    results = {}

    for student, answers in student_answers.items():
        student_result = {}
        for q_id, student_text in answers.items():
            ref_text = reference_answers[q_id]
            question_text = questions[q_id]

            # Compute SBERT similarity score
            similarity_score = compute_chunkwise_similarity(ref_text, student_text)

            # Get rubric scores + feedback from Mistral
            mistral_result = get_mistral_feedback_and_rubric(
                question_text, ref_text, student_text
            )

            combined_score = None
            if "rubric" in mistral_result:
                rubric_scores = mistral_result["rubric"]
                combined_score = combine_scores(similarity_score, rubric_scores)

            student_result[q_id] = {
                "similarity_score": similarity_score,
                "rubric_and_feedback": mistral_result,
                "combined_score": combined_score,
            }
        results[student] = student_result

    return results


# run the grading system
results = grade_long_answers(reference_answers, student_answers)

# Display the resultsfor student, answers in results.items():
for student, answers in results.items():
    print(f"\n📝 Results for {student}:")
    for q_id, info in answers.items():
        print(f"\nQuestion {q_id}:")
        print(f"  🔢 Similarity Score: {info['similarity_score']}")
        if "error" in info["rubric_and_feedback"]:
            print(f"  ⚠️ Error: {info['rubric_and_feedback']['error']}")
        else:
            rubric = info["rubric_and_feedback"]["rubric"]
            feedback = info["rubric_and_feedback"]["feedback"]
            print(f"  📊 Rubric Scores:")
            for crit, score in rubric.items():
                print(f"    - {crit}: {score}")
            print(f"  🧠 LLM Feedback:\n  {feedback}")
            if info.get("combined_score") is not None:
                print(f"  🎯 Combined Final Score: {info['combined_score']}")
