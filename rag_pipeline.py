"""
RAG Pipeline — LangChain + ChromaDB + HuggingFace Embeddings
Supports conversational Q&A over HR policy documents.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---- Paths ---------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
DATA_PATH   = str(BASE_DIR / "data")
CHROMA_PATH = str(BASE_DIR / "chroma_db")

# ---- Embedding model -----------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGPipeline:
    """
    Full RAG pipeline:
      1. Load & chunk HR policy documents
      2. Embed with all-MiniLM-L6-v2 → store in ChromaDB
      3. history-aware retriever  →  answer chain
    """

    def __init__(self):
        logger.info("Initialising RAG Pipeline …")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None
        self.rag_chain    = None
        self._initialise()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _initialise(self):
        """Load or build the vector store, then wire up the chain."""
        chroma_exists = (
            os.path.isdir(CHROMA_PATH)
            and any(Path(CHROMA_PATH).iterdir())
        )
        if chroma_exists:
            logger.info("Loading existing ChromaDB from %s", CHROMA_PATH)
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embeddings,
                collection_name="hr_policies",
            )
        else:
            logger.info("No ChromaDB found — ingesting documents …")
            self.ingest_documents()

        self._build_chain()

    # ------------------------------------------------------------------ #
    # Document Ingestion
    # ------------------------------------------------------------------ #

    def ingest_documents(self) -> int:
        """
        Load all .pdf files from DATA_PATH, chunk them, embed them,
        and persist to ChromaDB. Returns the number of chunks stored.
        """
        logger.info("Loading documents from %s", DATA_PATH)

        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents = loader.load()
        if not documents:
            raise FileNotFoundError(
                f"No .pdf documents found in {DATA_PATH}. "
                "Add HR policy PDF files and retry."
            )

        logger.info("Loaded %d documents. Splitting …", len(documents))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Created %d chunks.", len(chunks))

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_PATH,
            collection_name="hr_policies",
        )
        logger.info("ChromaDB persisted to %s", CHROMA_PATH)
        return len(chunks)

    # ------------------------------------------------------------------ #
    # Chain Construction
    # ------------------------------------------------------------------ #

    def _build_chain(self):
        """Build the history-aware retrieval + answer chain using LCEL."""
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10},
        )

        # --- Prompt 1: re-phrase the user question using chat history ---
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the chat history and the latest user question, "
             "rewrite the question to be fully self-contained. "
             "Do NOT answer it — just rephrase it if needed. "
             "If no rephrasing is needed, return it as-is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # --- Prompt 2: answer using retrieved context --------------------
        qa_system_prompt = (
            "You are HRBot, a knowledgeable and friendly HR assistant for "
            "Apex Technologies. Answer questions accurately based only on the "
            "provided HR policy context. "
            "If the answer is not in the context, say: "
            "'I don't have specific information on that in our HR policies. "
            "Please contact HR at hr@apextechnologies.com or call Ext. 1001.' "
            "Keep answers concise, clear, and professional. "
            "Format policy details as easy-to-read bullet points when helpful.\n\n"
            "Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        answer_chain  = create_stuff_documents_chain(llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(
            history_aware_retriever, answer_chain
        )

        logger.info("RAG chain ready.")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def chat(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
    ) -> dict:
        lc_history = []
        for human, ai in chat_history:
            lc_history.append(HumanMessage(content=human))
            lc_history.append(AIMessage(content=ai))

        result = self.rag_chain.invoke({
            "input": query,
            "chat_history": lc_history,
        })

        sources = []
        if "context" in result:
            sources = list({
                Path(doc.metadata.get("source", "HR Policy Manual")).name
                for doc in result["context"]
            })

        return {
            "answer":  result.get("answer", "I could not generate a response."),
            "sources": sources,
        }

    def get_collection_count(self) -> int:
        if self.vectorstore:
            return self.vectorstore._collection.count()
        return 0
