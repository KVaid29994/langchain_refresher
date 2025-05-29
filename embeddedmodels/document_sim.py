from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding= HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = ["Virat Kohli is Known for his aggressive batting and consistency, Kohli is one of the modern greats with over 25,000 international runs",
             "Rohit Sharma The 'Hitman' is famous for his effortless six-hitting and is the only player with three double centuries in ODIs.",
             "Jasprit Bumrah is India's pace spearhead, Bumrah is known for his deadly yorkers and unorthodox bowling action.",
             "Ravindra Jadeja is A true all-rounder, Jadeja contributes with bat, ball, and electric fielding across all formats."
             ,"KL Rahul is A stylish and versatile batter, Rahul can adapt to any format and has opened in all three formats for India."

]

query = "Tell me about Rohit Sharma"

document_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]
index, score =  (sorted((list(enumerate(scores))), key = lambda x:x[1]))[-1]
print (query)
print (documents[index])
print ("Simillary score is ", score)
