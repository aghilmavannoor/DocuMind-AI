import os
import re
import hashlib
from io import BytesIO

import fitz
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .source-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# GEMINI CLIENT
# =========================================================

if GEMINI_API_KEY:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

else:

    client = None


GEMINI_MODEL = "gemini-3.6-flash"


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# CONSTANTS
# =========================================================

CHUNK_SIZE = 600

SIM_THRESHOLD = 0.30

TOP_K_DEFAULT = 8


# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:

    st.session_state["chat_history"] = []


if "document_summary" not in st.session_state:

    st.session_state["document_summary"] = None


if "file_id" not in st.session_state:

    st.session_state["file_id"] = None


if "chunks" not in st.session_state:

    st.session_state["chunks"] = []


if "embeddings" not in st.session_state:

    st.session_state["embeddings"] = None


if "file_name" not in st.session_state:

    st.session_state["file_name"] = ""


if "num_pages" not in st.session_state:

    st.session_state["num_pages"] = 0


# =========================================================
# PDF FONT
# =========================================================

def register_pdf_font():

    possible_fonts = [

        r"C:\Windows\Fonts\arial.ttf",

        r"C:\Windows\Fonts\calibri.ttf",

        r"C:\Windows\Fonts\segoeui.ttf"

    ]

    for font_path in possible_fonts:

        if os.path.exists(font_path):

            try:

                pdfmetrics.registerFont(
                    TTFont(
                        "DocuMindFont",
                        font_path
                    )
                )

                return "DocuMindFont"

            except Exception:

                pass

    return "Helvetica"


PDF_FONT = register_pdf_font()


# =========================================================
# PDF TEXT CLEANING
# =========================================================

def clean_pdf_text(text):

    if not text:

        return ""

    text = text.replace(
        "\x00",
        ""
    )

    text = text.replace(
        "•",
        "-"
    )

    text = text.replace(
        "–",
        "-"
    )

    text = text.replace(
        "—",
        "-"
    )

    text = text.replace(
        "“",
        '"'
    )

    text = text.replace(
        "”",
        '"'
    )

    text = text.replace(
        "‘",
        "'"
    )

    text = text.replace(
        "’",
        "'"
    )

    return text


def escape_pdf_text(text):

    text = clean_pdf_text(
        str(text)
    )

    text = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return text


# =========================================================
# PAGE-BY-PAGE TEXT EXTRACTION
# =========================================================

def extract_pages(pdf_bytes):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text(
            "text"
        )

        text = text.strip()

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    document.close()

    return pages


# =========================================================
# SMART CHUNKING
# =========================================================

def create_chunks(pages):

    chunks = []

    chunk_id = 0

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        paragraphs = re.split(
            r"\n\s*\n",
            text
        )

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:

                continue

            if len(
                current_chunk
            ) + len(paragraph) + 1 <= CHUNK_SIZE:

                if current_chunk:

                    current_chunk += "\n\n"

                current_chunk += paragraph

            else:

                if len(
                    current_chunk.strip()
                ) > 80:

                    chunks.append(
                        {
                            "id": chunk_id,
                            "page": page_number,
                            "text": current_chunk.strip()
                        }
                    )

                    chunk_id += 1

                current_chunk = paragraph

        if len(
            current_chunk.strip()
        ) > 80:

            chunks.append(
                {
                    "id": chunk_id,
                    "page": page_number,
                    "text": current_chunk.strip()
                }
            )

            chunk_id += 1

    return chunks


# =========================================================
# EMBEDDINGS
# =========================================================

def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    if not texts:

        return np.array([])

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(
        embeddings
    )


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(
    question
):

    words = re.findall(
        r"\b[a-zA-Z]{2,}\b",
        question.lower()
    )

    stopwords = {
        "what",
        "when",
        "where",
        "which",
        "with",
        "from",
        "about",
        "does",
        "did",
        "this",
        "that",
        "have",
        "has",
        "there",
        "their",
        "they",
        "them",
        "into",
        "would",
        "could",
        "should",
        "the",
        "and",
        "are",
        "was",
        "were",
        "how",
        "why",
        "who",
        "his",
        "her",
        "he",
        "she",
        "it",
        "its",
        "is",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "do",
        "know",
        "list",
        "tell",
        "give",
        "mention",
        "mentioned",
        "document",
        "book",
        "page"
    }

    keywords = [
        word
        for word in words
        if word not in stopwords
    ]

    return list(
        dict.fromkeys(
            keywords
        )
    )


# =========================================================
# KEYWORD SCORE
# =========================================================

def keyword_score(
    question,
    text
):

    keywords = extract_keywords(
        question
    )

    if not keywords:

        return 0.0

    text_lower = text.lower()

    matches = 0

    for keyword in keywords:

        # Exact word matching is more reliable
        # than simple substring matching.
        pattern = (
            r"\b"
            + re.escape(keyword)
            + r"\b"
        )

        if re.search(
            pattern,
            text_lower
        ):

            matches += 1

    return (
        matches
        / len(keywords)
    )


# =========================================================
# EXACT PHRASE SCORE
# =========================================================

def exact_phrase_score(
    question,
    text
):

    question = question.lower().strip()

    text = text.lower()

    if len(question) < 5:

        return 0.0

    if question in text:

        return 1.0

    return 0.0


# =========================================================
# CONVERSATION CONTEXT
# =========================================================

def build_conversation_context():

    history = st.session_state.get(
        "chat_history",
        []
    )

    if not history:

        return ""

    recent_history = history[-4:]

    context = []

    for item in recent_history:

        context.append(
            f"User: {item['question']}"
        )

        context.append(
            f"Assistant: {item['answer']}"
        )

    return "\n".join(
        context
    )


# =========================================================
# CONTEXT-AWARE RETRIEVAL QUERY
# =========================================================

def build_retrieval_query(
    question
):

    history = st.session_state.get(
        "chat_history",
        []
    )

    if not history:

        return question

    recent_questions = [
        item["question"]
        for item in history[-3:]
    ]

    return (
        "Previous questions: "
        + " | ".join(recent_questions)
        + "\nCurrent question: "
        + question
    )


# =========================================================
# HYBRID RETRIEVAL
# =========================================================

def retrieve_chunks(
    question,
    chunks,
    embeddings,
    top_k
):

    if (
        not chunks
        or embeddings is None
        or len(embeddings) == 0
    ):

        return []

    retrieval_query = build_retrieval_query(
        question
    )

    query_embedding = embedding_model.encode(
        retrieval_query,
        normalize_embeddings=True
    )

    semantic_scores = np.dot(
        embeddings,
        query_embedding
    )

    keywords = extract_keywords(
        question
    )

    results = []

    for index, chunk in enumerate(
        chunks
    ):

        text = chunk["text"]
        text_lower = text.lower()

        semantic = float(
            semantic_scores[index]
        )

        keyword = keyword_score(
            question,
            text
        )

        exact = exact_phrase_score(
            question,
            text
        )

        # -------------------------------------------------
        # Direct keyword coverage
        # -------------------------------------------------

        matched_keywords = []

        for keyword_word in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword_word)
                + r"\b"
            )

            if re.search(
                pattern,
                text_lower
            ):

                matched_keywords.append(
                    keyword_word
                )

        direct_keyword_bonus = 0.0

        if matched_keywords:

            direct_keyword_bonus = min(
                0.20,
                0.05 * len(
                    matched_keywords
                )
            )

        # -------------------------------------------------
        # Section/title relevance
        # -------------------------------------------------

        section_bonus = 0.0

        section_terms = {
            "language": [
                "language",
                "languages"
            ],
            "qualification": [
                "qualification",
                "education",
                "degree",
                "academic",
                "university",
                "master",
                "bachelor"
            ],
            "internship": [
                "internship",
                "intern",
                "experience",
                "employment"
            ],
            "project": [
                "project",
                "projects"
            ],
            "skill": [
                "skill",
                "skills",
                "technical"
            ],
            "certification": [
                "certification",
                "certifications"
            ]
        }

        question_lower = question.lower()

        for terms in section_terms.values():

            question_has_section_term = any(
                term in question_lower
                for term in terms
            )

            text_has_section_term = any(
                re.search(
                    r"\b"
                    + re.escape(term)
                    + r"\b",
                    text_lower
                )
                for term in terms
            )

            if (
                question_has_section_term
                and text_has_section_term
            ):

                section_bonus = max(
                    section_bonus,
                    0.20
                )

        # -------------------------------------------------
        # Final hybrid score
        # -------------------------------------------------

        hybrid = (
            0.65 * semantic
            + 0.25 * keyword
            + direct_keyword_bonus
            + section_bonus
        )

        if exact > 0:

            hybrid = max(
                hybrid,
                0.90
            )

        results.append(
            {
                "chunk": chunk,
                "semantic": semantic,
                "keyword": keyword,
                "exact": exact,
                "hybrid": hybrid
            }
        )

    # Highest hybrid relevance first.
    #
    # IMPORTANT:
    # Do NOT filter out low-semantic/high-keyword chunks here.
    # Short factual questions such as "languages he know" can
    # have weak embedding similarity but an exact match to the
    # correct section of the document.
    #
    # The hybrid score already combines semantic + keyword +
    # direct keyword + section relevance, so it should decide
    # the final ranking.
    results.sort(
        key=lambda x: x["hybrid"],
        reverse=True
    )

    return results[:min(
        top_k,
        len(results)
    )]


# =========================================================
# BUILD GEMINI CONTEXT
# =========================================================

def build_gemini_context(
    retrieved_chunks
):

    context_parts = []

    for result in retrieved_chunks:

        chunk = result["chunk"]

        page = chunk["page"]

        text = chunk["text"]

        context_parts.append(
            f"[Page {page}]\n{text}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# ASK GEMINI
# =========================================================

def ask_gemini(
    question,
    context
):

    if not client:

        return (
            "Gemini API key is missing. "
            "Please add GEMINI_API_KEY "
            "to your .env file."
        )

    conversation_context = (
        build_conversation_context()
    )

    prompt = f"""
You are DocuMind AI, a document research assistant.

Answer the user's question ONLY using the provided document context.

Important rules:

1. Do not use outside knowledge.
2. If the answer is not supported by the context, say that the document context does not provide enough information.
3. Use the previous conversation only to understand references such as "it", "this", "that", or "its".
4. Do not invent page numbers.
5. Cite the actual page labels provided in the context.
6. Use citations like [Page 12].
7. Give a clear and useful answer.
8. Use bullet points when appropriate.

Previous conversation:

{conversation_context}

Document context:

{context}

Current question:

{question}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# =========================================================
# SUMMARY CONTEXT
# =========================================================

def build_summary_context(
    chunks
):

    if not chunks:

        return ""

    max_chunks = min(
        40,
        len(chunks)
    )

    indexes = np.linspace(
        0,
        len(chunks) - 1,
        max_chunks,
        dtype=int
    )

    selected = []

    seen = set()

    for index in indexes:

        index = int(index)

        if index in seen:

            continue

        seen.add(index)

        chunk = chunks[index]

        selected.append(
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(
        selected
    )


# =========================================================
# GENERATE DOCUMENT SUMMARY
# =========================================================

def generate_document_summary(
    chunks
):

    if not client:

        return (
            "Gemini API key is missing."
        )

    context = build_summary_context(
        chunks
    )

    prompt = f"""
You are DocuMind AI.

Create a structured summary of the uploaded document using ONLY the supplied context.

Do not use outside knowledge.

Your response MUST contain exactly these sections:

## 📋 Summary

Write a clear overall summary.

## 🔑 Key Insights

Provide the most important insights as bullet points.

## 🧩 Main Topics

List the major topics covered in the document.

## 📄 Important Pages

List important pages and briefly explain why each page matters.

Use page citations exactly like [Page 12].

Do not invent page numbers.

Document context:

{context}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# =========================================================
# PDF MARKDOWN PARSER
# =========================================================

def add_text_to_pdf(
    story,
    text,
    styles
):

    if not text:

        return

    lines = text.split(
        "\n"
    )

    for line in lines:

        line = line.strip()

        if not line:

            story.append(
                Spacer(1, 7)
            )

            continue

        line = clean_pdf_text(
            line
        )

        if line.startswith(
            "## "
        ):

            text_value = line[3:]

            story.append(
                Paragraph(
                    escape_pdf_text(
                        text_value
                    ),
                    styles["Heading2"]
                )
            )

            story.append(
                Spacer(1, 6)
            )

        elif line.startswith(
            "### "
        ):

            text_value = line[4:]

            story.append(
                Paragraph(
                    escape_pdf_text(
                        text_value
                    ),
                    styles["Heading3"]
                )
            )

            story.append(
                Spacer(1, 5)
            )

        elif line.startswith(
            "- "
        ):

            text_value = line[2:]

            story.append(
                Paragraph(
                    "• "
                    + escape_pdf_text(
                        text_value
                    ),
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(1, 4)
            )

        else:

            # Basic markdown cleanup
            line = line.replace(
                "**",
                ""
            )

            line = line.replace(
                "__",
                ""
            )

            story.append(
                Paragraph(
                    escape_pdf_text(
                        line
                    ),
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(1, 5)
            )


# =========================================================
# BUILD SUMMARY PDF
# =========================================================

def build_summary_pdf():

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    styles["Title"].fontName = PDF_FONT
    styles["Heading2"].fontName = PDF_FONT
    styles["Heading3"].fontName = PDF_FONT
    styles["BodyText"].fontName = PDF_FONT

    styles["Title"].alignment = TA_CENTER

    story = []

    file_name = st.session_state.get(
        "file_name",
        "Uploaded PDF"
    )

    num_pages = st.session_state.get(
        "num_pages",
        0
    )

    chunks = st.session_state.get(
        "chunks",
        []
    )

    # Title
    story.append(
        Paragraph(
            "DocuMind AI",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Document Summary Report",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Document:</b> "
            f"{escape_pdf_text(file_name)}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        Paragraph(
            f"<b>Pages:</b> {num_pages}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        Paragraph(
            f"<b>Text Chunks:</b> "
            f"{len(chunks)}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Document Summary",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    summary = st.session_state.get(
        "document_summary",
        ""
    )

    add_text_to_pdf(
        story,
        summary,
        styles
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Generated by DocuMind AI",
            styles["BodyText"]
        )
    )

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# BUILD Q&A PDF
# =========================================================

def build_qa_pdf():

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    styles["Title"].fontName = PDF_FONT
    styles["Heading2"].fontName = PDF_FONT
    styles["Heading3"].fontName = PDF_FONT
    styles["BodyText"].fontName = PDF_FONT

    styles["Title"].alignment = TA_CENTER

    story = []

    file_name = st.session_state.get(
        "file_name",
        "Uploaded PDF"
    )

    num_pages = st.session_state.get(
        "num_pages",
        0
    )

    # =====================================================
    # TITLE
    # =====================================================

    story.append(
        Paragraph(
            "DocuMind AI",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Document Research & Q&A Report",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            f"<b>Document:</b> "
            f"{escape_pdf_text(file_name)}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        Paragraph(
            f"<b>Pages:</b> {num_pages}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = st.session_state.get(
        "document_summary",
        ""
    )

    if summary:

        story.append(
            Paragraph(
                "Document Summary",
                styles["Heading2"]
            )
        )

        story.append(
            Spacer(1, 10)
        )

        add_text_to_pdf(
            story,
            summary,
            styles
        )

    # =====================================================
    # Q&A
    # =====================================================

    history = st.session_state.get(
        "chat_history",
        []
    )

    if history:

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Questions & Answers",
                styles["Heading2"]
            )
        )

        story.append(
            Spacer(1, 15)
        )

        for number, item in enumerate(
            history,
            start=1
        ):

            question = clean_pdf_text(
                item.get(
                    "question",
                    ""
                )
            )

            answer = item.get(
                "answer",
                ""
            )

            story.append(
                Paragraph(
                    f"Question {number}",
                    styles["Heading3"]
                )
            )

            story.append(
                Spacer(1, 5)
            )

            story.append(
                Paragraph(
                    "<b>Q:</b> "
                    + escape_pdf_text(
                        question
                    ),
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(1, 8)
            )

            story.append(
                Paragraph(
                    "<b>A:</b>",
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(1, 5)
            )

            add_text_to_pdf(
                story,
                answer,
                styles
            )

            story.append(
                Spacer(1, 15)
            )

    else:

        story.append(
            Spacer(1, 20)
        )

        story.append(
            Paragraph(
                "No questions have been asked yet.",
                styles["BodyText"]
            )
        )

    # =====================================================
    # FOOTER
    # =====================================================

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Generated by DocuMind AI - "
            "RAG-powered PDF research assistant.",
            styles["BodyText"]
        )
    )

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )

    top_k = st.slider(
        "Top K Results",
        min_value=3,
        max_value=15,
        value=TOP_K_DEFAULT
    )

    retrieval_only = st.checkbox(
        "🔍 Retrieval Only"
    )

    st.divider()

    st.header(
        "📄 Document Tools"
    )

    document_loaded = bool(
        st.session_state.get(
            "chunks"
        )
    )

    # =====================================================
    # GENERATE SUMMARY
    # =====================================================

    if st.button(
        "📋 Generate Summary",
        use_container_width=True,
        disabled=not document_loaded
    ):

        with st.spinner(
            "Generating document summary..."
        ):

            try:

                summary = generate_document_summary(
                    st.session_state[
                        "chunks"
                    ]
                )

                st.session_state[
                    "document_summary"
                ] = summary

                st.rerun()

            except Exception as e:

                st.error(
                    f"Summary generation failed: {e}"
                )

    # =====================================================
    # REGENERATE SUMMARY
    # =====================================================

    if st.button(
        "🔄 Regenerate Summary",
        use_container_width=True,
        disabled=not document_loaded
    ):

        with st.spinner(
            "Regenerating document summary..."
        ):

            try:

                summary = generate_document_summary(
                    st.session_state[
                        "chunks"
                    ]
                )

                st.session_state[
                    "document_summary"
                ] = summary

                st.rerun()

            except Exception as e:

                st.error(
                    f"Summary generation failed: {e}"
                )

    st.divider()

    # =====================================================
    # CLEAR CONVERSATION
    # =====================================================

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state[
            "chat_history"
        ] = []

        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧠 DocuMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered PDF research assistant using Hybrid RAG'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD PDF
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload a PDF document",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    file_id = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    if (
        st.session_state.get(
            "file_id"
        )
        != file_id
    ):

        with st.spinner(
            "Processing document..."
        ):

            pages = extract_pages(
                pdf_bytes
            )

            chunks = create_chunks(
                pages
            )

            embeddings = create_embeddings(
                chunks
            )

            st.session_state[
                "file_id"
            ] = file_id

            st.session_state[
                "file_name"
            ] = uploaded_file.name

            st.session_state[
                "num_pages"
            ] = len(pages)

            st.session_state[
                "chunks"
            ] = chunks

            st.session_state[
                "embeddings"
            ] = embeddings

            st.session_state[
                "document_summary"
            ] = None

            st.session_state[
                "chat_history"
            ] = []

        st.success(
            "PDF processed successfully!"
        )

        st.rerun()


# =========================================================
# DOCUMENT OVERVIEW
# =========================================================

if st.session_state.get(
    "chunks"
):

    st.header(
        "📊 Document Overview"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "📄 Pages",
            st.session_state.get(
                "num_pages",
                0
            )
        )

    with col2:

        st.metric(
            "🧩 Text Chunks",
            len(
                st.session_state.get(
                    "chunks",
                    []
                )
            )
        )

    with col3:

        st.metric(
            "📐 Embedding Size",
            384
        )

    st.divider()

# =========================================================
# SHOW SUMMARY
# =========================================================

if st.session_state.get(
    "document_summary"
):

    st.header(
        "📋 Document Summary"
    )

    st.markdown(
        st.session_state[
            "document_summary"
        ]
    )

    st.divider()


# =========================================================
# EXPORT REPORT
# =========================================================

has_report_content = (
    st.session_state.get(
        "document_summary"
    )
    or
    st.session_state.get(
        "chat_history"
    )
)

if has_report_content:

    st.header(
        "📥 Export Report"
    )

    st.write(
        "Download your DocuMind AI results "
        "as PDF reports."
    )

    summary_pdf = build_summary_pdf()

    qa_pdf = build_qa_pdf()

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.download_button(
            label="📋 Download Summary PDF",
            data=summary_pdf,
            file_name="documind_summary.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col2:

        st.download_button(
            label="💬 Download Q&A PDF",
            data=qa_pdf,
            file_name="documind_qa_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.divider()


# =========================================================
# CHAT SECTION
# =========================================================

if st.session_state.get(
    "chunks"
):

    st.header(
        "💬 Ask Your Document"
    )

    # Display previous conversation
    for item in st.session_state[
        "chat_history"
    ]:

        with st.chat_message(
            "user"
        ):

            st.write(
                item["question"]
            )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                item["answer"]
            )

    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        # =================================================
        # RETRIEVAL
        # =================================================

        retrieved = retrieve_chunks(
            question,
            st.session_state[
                "chunks"
            ],
            st.session_state[
                "embeddings"
            ],
            top_k
        )

        # =================================================
        # NO RESULTS
        # =================================================

        if not retrieved:

            answer = (
                "I could not find enough relevant "
                "information in the document to answer "
                "this question."
            )

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    answer
                )

            st.session_state[
                "chat_history"
            ].append(
                {
                    "question": question,
                    "answer": answer
                }
            )

            st.rerun()

        # =================================================
        # RETRIEVAL ONLY
        # =================================================

        elif retrieval_only:

            with st.chat_message(
                "assistant"
            ):

                st.subheader(
                    "🔍 Retrieved Sources"
                )

                for rank, result in enumerate(
                    retrieved,
                    start=1
                ):

                    chunk = result[
                        "chunk"
                    ]

                    st.markdown(
                        f"""
                        **#{rank} — Page {chunk['page']}**

                        Semantic: `{result['semantic']:.4f}`

                        Keyword: `{result['keyword']:.4f}`

                        Hybrid: `{result['hybrid']:.4f}`
                        """
                    )

                    with st.expander(
                        "View retrieved text"
                    ):

                        st.write(
                            chunk["text"]
                        )

        # =================================================
        # GEMINI ANSWER
        # =================================================

        else:

            context = build_gemini_context(
                retrieved
            )

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Thinking..."
                ):

                    try:

                        answer = ask_gemini(
                            question,
                            context
                        )

                        st.markdown(
                            answer
                        )

                    except Exception as e:

                        answer = (
                            f"Gemini request failed: {e}"
                        )

                        st.error(
                            answer
                        )

            # Save conversation
            st.session_state[
                "chat_history"
            ].append(
                {
                    "question": question,
                    "answer": answer
                }
            )

            # =================================================
            # RETRIEVAL DETAILS
            # =================================================

            with st.expander(
                "🔎 Retrieval Details"
            ):

                for rank, result in enumerate(
                    retrieved,
                    start=1
                ):

                    chunk = result[
                        "chunk"
                    ]

                    st.markdown(
                        f"""
                        **#{rank} — Page {chunk['page']}**

                        Semantic Score:
                        `{result['semantic']:.4f}`

                        Keyword Score:
                        `{result['keyword']:.4f}`

                        Hybrid Score:
                        `{result['hybrid']:.4f}`
                        """
                    )

            # =================================================
            # SOURCE TEXT
            # =================================================

            with st.expander(
                "📚 Source Text"
            ):

                for rank, result in enumerate(
                    retrieved,
                    start=1
                ):

                    chunk = result[
                        "chunk"
                    ]

                    st.markdown(
                        f"### Source {rank} — Page {chunk['page']}"
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.divider()

            # =================================================
            # ORIGINAL PDF PAGES
            # =================================================

            with st.expander(
                "📄 View Original PDF Pages"
            ):

                try:

                    pdf_document = fitz.open(
                        stream=pdf_bytes,
                        filetype="pdf"
                    )

                    shown_pages = set()

                    for result in retrieved:

                        page_number = result[
                            "chunk"
                        ]["page"]

                        if page_number in shown_pages:

                            continue

                        shown_pages.add(
                            page_number
                        )

                        page = pdf_document[
                            page_number - 1
                        ]

                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(
                                1.5,
                                1.5
                            )
                        )

                        image_bytes = pix.tobytes(
                            "png"
                        )

                        st.markdown(
                            f"### 📄 Page {page_number}"
                        )

                        st.image(
                            image_bytes,
                            use_container_width=True
                        )

                    pdf_document.close()

                except Exception as e:

                    st.error(
                        f"Could not render PDF pages: {e}"
                    )


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.info(
        "👆 Upload a PDF to start researching your document."
    )