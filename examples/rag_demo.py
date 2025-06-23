import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# 1. Example documents
example_docs = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space.",
    "Python is a popular programming language for AI.",
    "Chroma is an open-source vector database."
]

# 2. Split documents (optional for longer docs)
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = []
for doc in example_docs:
    docs.extend(text_splitter.create_documents([doc]))

# 3. Set up embeddings and vector store (using Sentence Transformers)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

# 4. Set up retriever and Gemini LLM
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="models/gemini-2.0-flash",
    temperature=0
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 5. User query loop
def main():
    print("Welcome to the RAG demo (Gemini)! Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == 'exit':
            break
        answer = qa_chain.run(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
        print("Please set your GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS environment variable.")
    else:
        main() 