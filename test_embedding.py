from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Rehabilitation helps people manage depression."

embedding = model.encode(text)

print(embedding)
print("Number of values:", len(embedding))