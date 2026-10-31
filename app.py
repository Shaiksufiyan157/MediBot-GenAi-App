from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from src.prompt import *
from dotenv import load_dotenv
import os


app=Flask(__name__)

load_dotenv()

OPENROUTER_API_KEY=os.environ.get("OPEN_ROUTER_API_KEY")
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')


# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

embeddings=download_hugging_face_embeddings()

index_name="medicalbot"

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

llm=ChatOpenRouter(    
    model="poolside/laguna-s-2.1",
    temperature=0.8,
    )

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)



question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response=rag_chain.invoke({"input":input})
    print(response)
    return str(response["answer"])
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)