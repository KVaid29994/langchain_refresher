from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

embedding = OpenAIEmbeddings('text-embedding-3-large', dimensions= 32)
result = embedding.embed_query("Delhi is the capital of India.")

print (str(result))