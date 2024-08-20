from flask import Flask, render_template, jsonify, request
from langchain_community.llms import LlamaCpp
from src.helper import download_huggingface_embedding
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = download_huggingface_embedding()

#Step 4: Initilise Pine cone and store chunks into Vector DB using the index created.
index_name = "med-bot"
docs = PineconeVectorStore.from_existing_index(index_name, embeddings)

prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": prompt}

#Using quantized version of llama3.1 Instruct model
llm = LlamaCpp(model_path="model/Meta-Llama-3.1-8B-Instruct--q4_0.bin",
                max_tokens=200,
                temperature=0.8,
            )

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docs.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")

def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(debug = True)