from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
genai.configure(api_key="your_google_api_key")

# loading and splitting the pdf files
pdf_paths = ["Attention_is_All_You_Need.pdf", "BERT.pdf", "GPT_3.pdf", "language_image_pretraining_with_knowledge_graphs.pdf", "LLaMA.pdf"]
docs=[]
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.split_documents(docs)
texts = [doc.page_content for doc in chunks]

# embedding and vectorstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(texts,embedding_model)

with open("faiss.pkl","wb") as f:
    pickle.dump(vector_store, f)

# retrieval and generation
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

retriever = vector_store.as_retriever(search_kwargs={"k":6})
rag_chain = RetrievalQA.from_chain_type(
    llm = ChatGoogleGenerativeAI(temperature=0,model="models/gemini-2.5-pro",google_api_key="your_google_api_key"),
    retriever = retriever,
    return_source_documents=True
)

# web interface 
from flask import Flask, render_template, request
app = Flask(__name__)
@app.route("/", methods=["GET","POST"])
def index():
    answer = ""
    retrieved_chunks = []
    if request.method == "POST":
        query = request.form["query"]
        docs_with_scores = vector_store.similarity_search_with_score(query, k=6)
        for doc, score in docs_with_scores:
            print(f"Similarity Score: {score}")
            print(f"Chunk Preview: {doc.page_content[:200]}...\n")
            retrieved_chunks.append((score, doc.page_content[:200]))
        result = rag_chain(query)
        answer = result["result"]
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)


