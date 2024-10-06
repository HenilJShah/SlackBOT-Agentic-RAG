from typing import List, Dict
from flask import Flask, request, jsonify
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
import json

from transformers import Tool, ReactJsonAgent

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
pdf_folder = "./docs"

# Global variable to store the singleton instance of vectordb
vectordb = None


class VectorDBSingleton:
    _instance = None

    def __new__(cls, documents):
        if cls._instance is None:
            print("Creating new VectorDB instance")
            embeddings = OpenAIEmbeddings()
            cls._instance = Chroma.from_documents(
                documents, embeddings, persist_directory="./vdb"
            )
        return cls._instance


from transformers.agents.llm_engine import MessageRole, get_clean_message_list

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class OpenAIEngine:

    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = client

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(
            messages, role_conversions=openai_role_conversions
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
        )
        return response.choices[0].message.content


def load_pdf_files(pdf_folder):
    """Load PDF files from a folder, return document objects and extracted text."""
    documents = []
    texts = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            try:
                pdf_loader = PyMuPDFLoader(file_path=pdf_path)
                pdf_docs = pdf_loader.load()
                documents.extend(pdf_docs)
                texts.extend([doc.page_content for doc in pdf_docs])
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return documents, texts


def split_text_into_chunks(text):
    """Split the extracted text into chunks using text splitting."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n", "\n\n"]
    )
    return text_splitter.split_text(text)


def wrap_text_in_documents(texts):
    """Wrap each text chunk in a Document object."""
    return [Document(page_content=text) for text in texts]


class AnswerSchema(BaseModel):
    question: str = Field(description="The question asked")
    answer: str = Field(description="The answer extracted from the context")


parser = JsonOutputParser(pydantic_object=AnswerSchema)

# Define the PromptTemplate
template = """You are a helpful assistant. Use the provided context from the document to answer the question accurately. If the answer is not available, respond with 'Data Not Available'.

Context:
{context}

{format_instructions}

Question:
{query}
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def get_answer_from_context(query, context_chunks):
    """Format the prompt and fetch the answer using OpenAI."""
    context = "\n".join([chunk.page_content for chunk, _ in context_chunks])
    formatted_prompt = prompt.format(context=context, query=query)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt},
        ],
    )
    return json.dumps(response.choices[0].message.content)


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"

    def __init__(self, vectordb, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [
                f"===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


def get_answers_for_questions(
    questions: List[str], vectordb, k: int = 3
) -> Dict[str, str]:
    """Fetch answers for the provided questions using the vector database."""
    answers = {}
    for question in questions:
        context_chunks = vectordb.similarity_search_with_score(question, k=k)
        answer = get_answer_from_context(question, context_chunks)
        answers[question] = answer
    return answers


def run_agentic_rag(question: str) -> str:
    enhanced_question = f"""
    Using the information contained in your knowledge base, which you can access with the 'retriever' tool, give a comprehensive answer to the question below. Respond only to the question asked, response should be concise and relevant to the question. If you cannot find information, do not give up and try calling your retriever again with different arguments! Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries. Your queries should not be questions but affirmative form sentences: e.g. rather than  "query should be "What is the termination policy?", "What is their vacation policy?".

    Question:
    {question}
"""
    retriever_tool = RetrieverTool(vectordb)
    llm_engine = OpenAIEngine()

    agent = ReactJsonAgent(
        tools=[retriever_tool], llm_engine=llm_engine, max_iterations=5, verbose=2
    )

    return agent.run(enhanced_question)


# Flask API endpoint for the AI agent
@app.route("/ask", methods=["POST"])
def ask_question():
    global vectordb

    # Check if vectordb is already initialized, if not load it
    if vectordb is None:
        # Load PDF files
        pdf_docs, pdf_texts = load_pdf_files(pdf_folder=pdf_folder)
        # Initialize ChromaDB (singleton pattern)
        documents = wrap_text_in_documents(pdf_texts)
        vectordb = VectorDBSingleton(documents)

    # Get question from request
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        # Search for context chunks
        # context_chunks = vectordb.similarity_search_with_score(question, k=3)
        # Get the answer
        # RAG based answer
        # answer = get_answer_from_context(question, context_chunks)

        # agent based answer
        answer = run_agentic_rag(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return "ask bot"


@app.route("/", methods=["GET"])
def home():
    return "bot up"


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
