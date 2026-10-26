"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context can contain FAQ entries, resolved support tickets, and telecom guide pages.

CITATION RULES:
- Cite every factual instruction, policy, or resolution inline in the answer.
- Copy source labels exactly as shown in context, e.g. [FAQ 1], [Ticket TK-001], or [Guide p. 4].
- Place citations at the end of the sentence they support. Use multiple citations when sources support a sentence.
- Never invent a ticket ID, FAQ ID, or page number. If the context does not support a claim, do not make that claim.
- Do not cite anything that is not present in the context.

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        metadata = doc.metadata
        source = str(metadata.get("source", "unknown")).lower()
        if source == "faq":
            citation = f"FAQ {metadata.get('faq_id', 'unknown')}"
        elif source == "ticket":
            citation = f"Ticket {metadata.get('ticket_id', 'unknown')}"
        elif source == "guide":
            page = metadata.get("page")
            try:
                page = int(page) + 1
                citation = f"Guide p. {page}"
            except (TypeError, ValueError):
                citation = "Guide page unknown"
        else:
            citation = "Unknown source"
        sections.append(f"[{citation}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Set it in the repository .env file and restart."
        )

    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    prepare = RunnableLambda(
        lambda question: {"question": question, "docs": retriever.invoke(question)}
    )
    answer = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda values: _format_docs(values["docs"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return prepare | RunnableBranch(
        (
            lambda values: not values["docs"],
            RunnableLambda(lambda _: "I don't know based on the available support information. Please call 611 or use the MyTelecom app."),
        ),
        answer,
    )
