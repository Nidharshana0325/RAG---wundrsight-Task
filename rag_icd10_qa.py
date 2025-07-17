import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_document(pdf_path):
    """Load the ICD-10 PDF document."""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} pages from the PDF.")
        return documents
    except Exception as e:
        logging.error(f"Failed to load PDF: {str(e)}")
        raise

def split_documents(documents):
    """Split documents into 500-token chunks with 50-token overlap."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logging.error(f"Failed to split documents: {str(e)}")
        raise

def create_vector_store(chunks):
    """Generate embeddings using all-MiniLM-L6-v2 and store in FAISS."""
    try:
        
        if "HF_TOKEN" in os.environ:
            os.environ.pop("HF_TOKEN")
            
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        logging.info("Vector store created with FAISS.")
        return vector_store
    except Exception as e:
        logging.error(f"Failed to create vector store: {str(e)}")
        raise


def setup_llm():
    """Initialize the LLM using facebook/opt-350m for lightweight execution."""
    try:
        model_name = "facebook/opt-350m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.3,  
            do_sample=True
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("LLM initialized.")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {str(e)}")
        raise


def create_rag_pipeline(vector_store, llm):
    """Create the RAG pipeline using LangChain's RetrievalQA."""
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            verbose=True 
        )
        logging.info("RAG pipeline created.")
        return qa_chain
    except Exception as e:
        logging.error(f"Failed to create RAG pipeline: {str(e)}")
        raise


def query_rag(qa_chain, query):
    """Run a query through the RAG pipeline and return the answer and source documents."""
    try:
        result = qa_chain({"query": query})
        answer = result.get("result", "No answer found")
        sources = result.get("source_documents", [])
        return answer, sources
    except Exception as e:
        logging.error(f"Failed to run query: {str(e)}")
        raise

def main():
    try:
        
        pdf_path = "9241544228_eng.pdf"  
        query = "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission"

        
        documents = load_document(pdf_path)
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks)
        
        llm = setup_llm()
        qa_chain = create_rag_pipeline(vector_store, llm)
        
        
        answer, source_documents = query_rag(qa_chain, query)
        
        
        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")
        print("\nSource Documents:")
        for i, doc in enumerate(source_documents, 1):
            print(f"\nDocument {i}:")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...")
            
    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")

if __name__ == "__main__":
    main()