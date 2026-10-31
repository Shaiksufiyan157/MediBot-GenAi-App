from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from src.helper import download_hugging_face_embeddings,text_split,load_pdf_file
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)

extracted_data=load_pdf_file("data/")
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()
index_name = "medicalbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 


# Embed each chunk and upsert the embeddings into your Pinecone index.
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)