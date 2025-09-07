# import requests

# prompt = "Write a short feedback on this answer: Photosynthesis is the process plants use to make food."

# try:
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": "phi", "prompt": prompt, "stream": False},
#     )
#     data = response.json()
#     print("Full response JSON:", data)

#     # Try common keys:
#     feedback = data.get("response") or data.get("text")
#     if not feedback and "choices" in data and len(data["choices"]) > 0:
#         feedback = data["choices"][0].get("text", "")

#     print("Extracted feedback:", feedback.strip() if feedback else "No feedback found")

# except Exception as e:
#     print(f"Error: {e}")


# from gensim import corpora, models

# # Sample documents
# documents = [
#     "Human machine interface for lab abc computer applications",
#     "A survey of user opinion of computer system response time",
#     "The EPS user interface management system",
# ]

# # Preprocessing
# texts = [doc.lower().split() for doc in documents]
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]


# # print("Texts: ", texts)
# # print("Dictionary: ", dictionary)
# # print("Corpus: ", corpus)

# # Train LDA model
# lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# # View topics
# topics = lda_model.print_topics(num_words=4)
# for topic in topics:
#     print(topic)


import requests
import json

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
    },
    data=json.dumps(
        {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": "What is the meaning of life?"}],
        }
    ),
)

print(response.status_code)
print(response.text)
