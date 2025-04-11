from langchain_ollama import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

class RAGPipeline:
    def __init__(self,
                 llm_model: str = "mistral",  # ✅ Ollama mistral model name
                 embed_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 750,
                 chunk_overlap: int = 100,
                 chroma_db_dir: str = "chroma_store"):

        # ✅ Use local LLM via Ollama
        self.llm = ChatOllama(model=llm_model)

        self.embedding_model = HuggingFaceEmbeddings(model_name=embed_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_db_dir = chroma_db_dir
        self.raw_docs = None
        self.full_text = ""
        self.enriched_chunks = None
        self.vectordb = None
        self.qa_chain = None

    def load_pdf(self, path: str):
        loader = PyMuPDFLoader(path)
        self.raw_docs = loader.load()
        self.full_text = "\n".join([doc.page_content for doc in self.raw_docs])
        print(f"Loaded {len(self.raw_docs)} documents from {path}.")

    def chunk_documents(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(self.raw_docs)

    def add_context_to_chunks(self, chunks):
        enriched = []
        for i, chunk in enumerate(chunks):
            print(f"Contextualizing chunk {i+1}/{len(chunks)}...")
            doc_prompt = DOCUMENT_CONTEXT_PROMPT.format(doc_content=self.full_text[:500])
            chunk_prompt = CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk.page_content)
            full_prompt = doc_prompt + "\n" + chunk_prompt
            response = self.llm.invoke(full_prompt)
            new_text = f"{response.content.strip()}\n{chunk.page_content}"
            chunk.page_content = new_text
            enriched.append(chunk)
        self.enriched_chunks = enriched

    def store_in_chroma(self):
        self.vectordb = Chroma.from_documents(
            documents=self.enriched_chunks,
            embedding=self.embedding_model,
            persist_directory=self.chroma_db_dir
        )
        self.vectordb.persist()

    def build_retrieval_qa_chain(self):
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=False
        )

    def process_pdf(self, pdf_path: str):
        self.load_pdf(pdf_path)
        chunks = self.chunk_documents()
        self.add_context_to_chunks(chunks)
        self.store_in_chroma()
        self.build_retrieval_qa_chain()
        print("PDF processing complete and QA chain built.")

    def query(self, user_query: str):
        if not self.qa_chain:
            raise ValueError("The QA chain is not built. Run process_pdf() first.")
        return self.qa_chain.run(user_query)


if __name__ == "__main__":
    pipeline = RAGPipeline(
    llm_model="mistral",  
    embed_model="all-MiniLM-L6-v2",  # Or any SentenceTransformers model
    chunk_size=750,
    chunk_overlap=100,
    chroma_db_dir="chroma_store")

    pipeline.process_pdf("GSTsmartGuide.pdf")


    response = pipeline.query("Please Enter your query")
    print(response)