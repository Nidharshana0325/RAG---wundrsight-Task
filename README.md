
## Code Structure
- **File**: `rag_pipeline.py`
- **Functions**:
  - `load_document`: Loads the PDF using `PyPDFLoader`.
  - `split_documents`: Splits text into chunks using `RecursiveCharacterTextSplitter`.
  - `create_vector_store`: Generates embeddings and stores them in FAISS.
  - `setup_llm`: Initializes `opt-350m` via `HuggingFacePipeline`.
  - `create_rag_pipeline`: Sets up the `RetrievalQA` chain with MMR.
  - `query_rag`: Executes queries and returns answers with source documents.
  - `main`: Orchestrates the pipeline.
- **Modularity**: Functions are independent and reusable.
- **Logging**: Comprehensive logging for runtime feedback.
- **Error Handling**: Try-catch blocks ensure robust execution.

## Time Management
- **Total Time**: ~3.5 hours
  - Document loading and splitting: 30 minutes
  - Embedding and vector store setup: 45 minutes
  - LLM setup and pipeline integration: 1 hour
  - Testing and debugging: 1 hour
  - README writing: 30 minutes
- **Prioritization**: Focused on core pipeline to deliver a functional MVP within the 4-hour limit, skipping optional features to ensure completeness and quality.

## Future Improvements
- Add a Gradio/Streamlit interface for user-friendly interaction.
- Implement dynamic document uploads to support multiple PDFs.
- Add caching for repeat queries to improve performance.
- Enhance reranking with custom logic or models for better relevance.
etype: text/markdown
- Use a larger LLM (e.g., LLaMA 7B) for improved answer quality, if resources permit.
- Explore GPU-accelerated FAISS for faster retrieval.
