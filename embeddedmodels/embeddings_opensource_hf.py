from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer
embedding= HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2' )
sentences = "Welcome to Delhi"
vector = embedding.embed_query(sentences)
print(vector)