"""Query with RAG using a local vector database and OpenAI's ChatGPT model."""
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from local_dir_rag.vector_store import load_vector_database
from local_dir_rag.text_processor import format_documents, print_sources

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(None)


def query_loop(vector_db_path=None, k: int = 10):
    """
    Run an interactive RAG-based chat session using a local vector database
    and OpenAI's ChatGPT model.
    """

    # Load the vector database
    vector_db = load_vector_database(vector_db_path)
    logger.info("Vector database loaded successfully from %s", vector_db_path)

    # Set up the chat model
    chat_model = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7
    )

    # Create the RAG prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful assistant that provides accurate information based on
    the given context. If you don't know the answer based on the context,
    just say that you don't know. Don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # Create a retriever from the vector database
    retriever = vector_db.as_retriever(search_kwargs={"k": k})

    # Create the RAG chain
    rag_chain = (
        {
            "context": retriever | print_sources | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    # Interactive query loop
    print("Local RAG Chat Session")
    print("Type your questions below.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        prompt = input("\nPrompt: ")

        # Check for exit command
        if prompt.lower() in ['exit', 'quit']:
            print("Exiting chat session.")
            break

        # Generate answer
        prompt = rag_chain.invoke(prompt)
        print(f"\nResponse: {prompt}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    load_dotenv(dotenv_path=".env.params")

    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
    query_loop(VECTOR_DB_PATH)
