from flask import Flask, request, render_template
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import os
import uuid

app = Flask(__name__)

# Model and embedding setup
MODEL = "llama2"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Prompt templates
template = """Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""

# Initialize Chroma with a local persistent client
client = chromadb.PersistentClient(path="chroma_db")

def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"Successfully loaded PDF with PyPDFLoader. Number of pages: {len(pages)}")
        return pages
    except Exception as e:
        print(f"PyPDFLoader failed: {str(e)}")
        try:
            loader = PDFMinerLoader(file_path)
            pages = loader.load()
            print(f"Successfully loaded PDF with PDFMinerLoader. Number of pages: {len(pages)}")
            return pages
        except Exception as e:
            print(f"PDFMinerLoader failed: {str(e)}")
            raise ValueError(f"Unable to load PDF: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            question = request.form.get("question")
            filename = file.filename
            file_path = os.path.join("uploads", filename)
            file.save(file_path)

            # Load PDF and split into chunks
            try:
                pages = load_pdf(file_path)
                print(f"PDF loaded successfully. Number of pages: {len(pages)}")
            except Exception as e:
                return f"Error loading PDF: {str(e)}"

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(pages)
            print(f"Number of text chunks after splitting: {len(texts)}")

            # Create a unique collection name for each upload
            collection_name = f"collection_{uuid.uuid4().hex[:10]}"
            print(f"Using collection name: {collection_name}")

            # Create Chroma vector store
            try:
                vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    client=client,
                    collection_name=collection_name
                )
                print(f"Vector store created successfully with {len(texts)} documents")
            except Exception as e:
                return f"Error creating vector store: {str(e)}"

            # Retrieve context
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 1000})  # Retrieve up to 1000 relevant chunks
                context = retriever.get_relevant_documents(question)
                print(f"Retrieved {len(context)} relevant documents")
                for i, doc in enumerate(context):
                    print(f"Document {i+1} content: {doc.page_content}")
            except Exception as e:
                return f"Error retrieving context: {str(e)}"

            # Join the context into a single string
            context_text = "\n\n".join([doc.page_content for doc in context])

            # Create prompt
            prompt = PromptTemplate.from_template(template)

            # Run the model and get the answer
            try:
                inputs = {"context": context_text, "question": question}
                chain = prompt.format(**inputs)
                answer = model.invoke(chain)

                # Format the answer with proper indentation and spacing
                formatted_answer = f"""
                {answer}
                """
                print(f"Generated answer: {formatted_answer}")
            except Exception as e:
                return f"Error generating answer: {str(e)}"

            return render_template("result.html", answer=formatted_answer, context=context_text)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)