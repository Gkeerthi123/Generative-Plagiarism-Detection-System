
import streamlit as st
import sys
import os
import re
import html
import numpy as np
import pandas as pd
import plotly.express as px

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_documents
from src.document_reader import extract_text
from src.text_preprocessing import clean_text
from src.embedding import load_embedding_model, generate_embeddings

from src.plagiarism_detector import (
    build_source_index,
    detect_plagiarism,
    find_top_matches
)

from src.semantic_search import (
    save_faiss_index,
    load_faiss_index
)

from src.reference_manager import (
    save_reference_files,
    load_reference_chunks
)

from src.text_matching import (
    find_exact_word_matches,
    find_phrase_matches,
    find_matching_passages
)

from src.pdf_report import generate_pdf_report
from src.ai_summary import generate_ai_summary


# --------------------------------
# HELPER FUNCTIONS
# --------------------------------

def highlight_shared_words(text, shared_words):

    escaped_text = html.escape(text)

    if not shared_words:
        return escaped_text

    pattern = re.compile(
        r"\b("
        + "|".join(
            re.escape(word)
            for word in sorted(
                shared_words,
                key=len,
                reverse=True
            )
        )
        + r")\b",
        re.IGNORECASE
    )

    return pattern.sub(
        lambda match:
        '<mark style="background-color:#fff176;color:#000;'
        'padding:2px;">'
        + match.group(0)
        + '</mark>',
        escaped_text
    )


def get_similarity_percent(score):

    score = float(score)

    # Keep displayed similarity between 0 and 100.
    score = max(0.0, min(1.0, score))

    return round(score * 100, 2)


def get_risk_level(score):

    if score >= 0.80:
        return "High"

    elif score >= 0.65:
        return "Medium"

    return "Low"


def get_source_name(match):

    return os.path.basename(
        str(
            match.get(
                "source_file",
                match.get(
                    "file_name",
                    "Unknown reference"
                )
            )
        )
    )


def remove_self_matches(results, uploaded_name):

    uploaded_name = os.path.basename(
        uploaded_name
    ).strip().lower()

    return [
        result
        for result in results
        if get_source_name(result).strip().lower()
        != uploaded_name
    ]


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Generative Plagiarism Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Generative Plagiarism Detection System")

st.write(
    "Upload a document to check for potentially similar "
    "content against PAN-PC-11 and your own reference documents."
)

st.info(
    "The system checks exact words, repeated phrases, "
    "and meaning-based similarities using AI."
)

st.warning(
    "Similarity scores are indicators. They do not automatically "
    "prove plagiarism."
)


custom_reference_folder = os.path.join(
    PROJECT_ROOT,
    "data",
    "custom_references"
)


# --------------------------------
# ADD REFERENCE DOCUMENTS
# --------------------------------

st.header("Add Your Reference Documents")

reference_files = st.file_uploader(
    "Upload original reference documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="reference_files"
)

if st.button("Add Reference Documents"):

    if reference_files:

        try:

            saved_files = save_reference_files(
                reference_files,
                custom_reference_folder
            )

            st.cache_resource.clear()

            st.success(
                "Added reference documents: "
                + ", ".join(saved_files)
            )

        except Exception as error:

            st.error(
                f"Unable to save references: {error}"
            )

    else:

        st.warning(
            "Please select reference documents."
        )


# --------------------------------
# LOAD MODEL AND REFERENCE DATABASE
# --------------------------------

@st.cache_resource(show_spinner=False)
def load_detection_resources():

    model = load_embedding_model()

    index_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "source_index.faiss"
    )

    chunks_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "source_chunks.pkl"
    )

    index, source_chunks = load_faiss_index(
        index_path,
        chunks_path
    )

    if index is None:

        source_folder = os.path.join(
            PROJECT_ROOT,
            "data",
            "raw",
            "pan_pc11",
            "source-document"
        )

        source_documents = load_documents(
            source_folder
        )

        # Use 10 PAN-PC-11 documents for faster indexing.
        source_documents = source_documents[:10]

        for document in source_documents:

            document["text"] = clean_text(
                document["text"]
            )

        index, source_chunks = build_source_index(
            source_documents,
            model
        )

        save_faiss_index(
            index,
            source_chunks,
            index_path,
            chunks_path
        )

    for chunk in source_chunks:

        chunk.setdefault(
            "source_type",
            "PAN-PC-11 Dataset"
        )

    # Load user-uploaded reference documents.
    custom_chunks = load_reference_chunks(
        custom_reference_folder
    )

    if custom_chunks:

        custom_texts = [
            chunk["text"]
            for chunk in custom_chunks
        ]

        custom_embeddings = generate_embeddings(
            custom_texts,
            model
        )

        index.add(
            np.asarray(
                custom_embeddings,
                dtype="float32"
            )
        )

        for chunk in custom_chunks:

            chunk["source_type"] = (
                "Custom Reference Document"
            )

        source_chunks.extend(
            custom_chunks
        )

    return model, index, source_chunks


# --------------------------------
# UPLOAD SUSPICIOUS DOCUMENT
# --------------------------------

st.header("Upload Your Document to Check")

input_method = st.radio(
    "Choose input method",
    ["Upload File", "Paste Text"],
    horizontal=True
)

document_text = ""
document_name = ""

if input_method == "Upload File":

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt"],
        key="suspicious_file"
    )

    if uploaded_file is not None:

        document_name = uploaded_file.name

        document_text = extract_text(
            document_name,
            uploaded_file.getvalue()
        )

        if document_text:

            st.success(
                f"Uploaded: {document_name}"
            )

        else:

            st.error(
                "Unable to extract readable text."
            )

else:

    document_name = "Pasted_Text.txt"

    document_text = st.text_area(
        "Paste your document content here",
        height=250
    )


# --------------------------------
# DOCUMENT PREVIEW
# --------------------------------

if document_text.strip():

    st.subheader("Document Preview")

    st.text_area(
        "Extracted document content",
        document_text[:2000],
        height=180,
        disabled=True
    )

    st.caption(
        f"Extracted characters: {len(document_text)}"
    )


# --------------------------------
# SIMILARITY THRESHOLD
# --------------------------------

threshold = st.slider(
    "Similarity threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.70,
    step=0.05
)


# --------------------------------
# CHECK FOR PLAGIARISM
# --------------------------------

if st.button(
    "Check for Plagiarism",
    type="primary"
):

    if not document_text.strip():

        st.warning(
            "Please upload a document or paste text."
        )

    else:

        with st.spinner(
            "Loading AI model and references..."
        ):

            try:

                model, index, source_chunks = (
                    load_detection_resources()
                )

            except Exception as error:

                st.error(
                    f"Resource loading failed: {error}"
                )

                st.stop()

        suspicious_clean_text = clean_text(
            document_text
        )

        suspicious_document = {
            "file_name": document_name,
            "text": suspicious_clean_text
        }

        with st.spinner(
            "Analyzing document..."
        ):

            try:

                matches = detect_plagiarism(
                    suspicious_document,
                    index,
                    source_chunks,
                    model,
                    threshold
                )

                top_matches = find_top_matches(
                    suspicious_document,
                    index,
                    source_chunks,
                    model,
                    top_k=5
                )

                matches = remove_self_matches(
                    matches,
                    document_name
                )

                top_matches = remove_self_matches(
                    top_matches,
                    document_name
                )

            except Exception as error:

                st.error(
                    f"Plagiarism analysis failed: {error}"
                )

                st.stop()

        # --------------------------------
        # DASHBOARD
        # --------------------------------

        st.header(
            "📊 Plagiarism Analysis Dashboard"
        )

        highest_similarity = max(
            (
                max(
                    0.0,
                    min(
                        1.0,
                        float(item["similarity_score"])
                    )
                )
                for item in top_matches
            ),
            default=0.0
        )

        risk_level = get_risk_level(
            highest_similarity
        )

        unique_matched_documents = len(
            set(
                get_source_name(item)
                for item in matches
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Highest Similarity",
                f"{get_similarity_percent(highest_similarity):.2f}%"
            )

        with col2:

            st.metric(
                "Matched Documents",
                unique_matched_documents
            )

        with col3:

            st.metric(
                "Risk Indicator",
                risk_level
            )

        if risk_level == "High":

            st.error(
                "High similarity detected. Review the matched passages."
            )

        elif risk_level == "Medium":

            st.warning(
                "Moderate similarity detected. Review the results."
            )

        else:

            st.success(
                "Low similarity detected in the closest matches."
            )

        st.caption(
            "Risk indicator is based on the highest score among "
            "the top five closest matches. High >= 80%, "
            "Medium >= 65%, Low below 65%. "
            "It is not a confirmed plagiarism verdict."
        )

        # --------------------------------
        # AI ANALYSIS REPORT
        # --------------------------------

        st.header(
            "🤖 AI-Powered Analysis Summary"
        )

        with st.spinner(
            "AI is generating your analysis report..."
        ):

            ai_summary = generate_ai_summary(
                document_name,
                highest_similarity,
                risk_level,
                matches
            )

        st.subheader("AI Analysis Report")

        st.markdown(
            ai_summary
        )

        # --------------------------------
        # SIMILARITY CHART
        # --------------------------------

        if top_matches:

            dashboard_data = pd.DataFrame([

                {
                    "Reference Document": get_source_name(item),

                    "Similarity (%)": get_similarity_percent(
                        item["similarity_score"]
                    )
                }

                for item in top_matches

            ])

            dashboard_chart = (
                dashboard_data
                .groupby("Reference Document")[
                    "Similarity (%)"
                ]
                .max()
                .reset_index()
            )

            st.subheader(
                "Reference Similarity Overview"
            )

            fig = px.bar(
                dashboard_chart,
                x="Reference Document",
                y="Similarity (%)",
                title="Similarity Score by Reference Document",
                text="Similarity (%)"
            )

            fig.update_yaxes(
                range=[0, 100],
                title="Similarity (%)"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                cliponaxis=False
            )

            fig.update_layout(
                yaxis=dict(
                    range=[0, 110],
                    title="Similarity (%)"
                ),
                xaxis_title="Reference Document",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # --------------------------------
        # SEMANTIC MATCHING RESULTS
        # --------------------------------

        st.header(
            "Plagiarism Detection Results"
        )

        if matches:

            st.warning(
                f"Found {len(matches)} potentially similar sections."
            )

            result_data = []

            for result in matches:

                result_data.append({

                    "Source Document": get_source_name(result),

                    "Reference Type": result.get(
                        "source_type",
                        "PAN-PC-11 Dataset"
                    ),

                    "Similarity (%)": get_similarity_percent(
                        result["similarity_score"]
                    )

                })

            results_df = pd.DataFrame(
                result_data
            )

            st.subheader(
                "Semantic Matching Sources"
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            st.subheader(
                "Semantic Match Details - Side-by-Side Comparison"
            )

            for number, result in enumerate(
                matches,
                start=1
            ):

                st.markdown(
                    f"### Match {number}: "
                    f"{get_source_name(result)}"
                )

                st.write(
                    f"Similarity: "
                    f"{get_similarity_percent(result['similarity_score']):.2f}%"
                )

                shared_words = find_exact_word_matches(
                    result["suspicious_text"],
                    result["matched_source_text"]
                )

                st.caption(
                    f"Number of shared meaningful words: "
                    f"{len(shared_words)}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**Uploaded Document**"
                    )

                    highlighted = highlight_shared_words(
                        result["suspicious_text"],
                        shared_words
                    )

                    st.markdown(
                        f"""
                        <div style="
                            padding:12px;
                            border:1px solid #999;
                            border-radius:8px;
                            line-height:1.8;
                            overflow-wrap:anywhere;
                        ">
                        {highlighted}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        "**Reference Document**"
                    )

                    highlighted = highlight_shared_words(
                        result["matched_source_text"],
                        shared_words
                    )

                    st.markdown(
                        f"""
                        <div style="
                            padding:12px;
                            border:1px solid #999;
                            border-radius:8px;
                            line-height:1.8;
                            overflow-wrap:anywhere;
                        ">
                        {highlighted}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.divider()

        else:

            st.info(
                "No semantic matches found above "
                "the selected threshold."
            )

        # --------------------------------
        # EXACT WORD AND PHRASE MATCHING
        # --------------------------------

        st.header(
            "Exact Word and Phrase Matching"
        )

        word_matches_by_source = {}
        phrase_matches_by_source = {}

        for source in source_chunks:

            source_name = source["file_name"]

            if (
                source_name.strip().lower()
                == document_name.strip().lower()
            ):
                continue

            source_text = source["text"]

            common_words = find_exact_word_matches(
                suspicious_clean_text,
                source_text
            )

            common_phrases = find_phrase_matches(
                suspicious_clean_text,
                source_text
            )

            if common_words:

                word_matches_by_source.setdefault(
                    source_name,
                    set()
                ).update(common_words)

            if common_phrases:

                phrase_matches_by_source.setdefault(
                    source_name,
                    set()
                ).update(common_phrases)

        if word_matches_by_source:

            st.subheader(
                "Shared Words"
            )

            for source_name, words in word_matches_by_source.items():

                st.write(
                    f"**Source:** {source_name}"
                )

                st.write(
                    ", ".join(sorted(words))
                )

        else:

            st.info(
                "No exact shared words found."
            )

        if phrase_matches_by_source:

            st.subheader(
                "Exact Phrase Matches"
            )

            for source_name, phrases in phrase_matches_by_source.items():

                st.write(
                    f"**Source:** {source_name}"
                )

                for phrase in sorted(phrases):

                    st.write(
                        f"- {phrase}"
                    )

        else:

            st.info(
                "No exact phrase matches found."
            )

        # --------------------------------
        # MATCHING PASSAGES
        # --------------------------------

        st.subheader(
            "Matching Passages"
        )

        passage_matches_by_source = {}

        for source in source_chunks:

            if (
                source["file_name"].strip().lower()
                == document_name.strip().lower()
            ):
                continue

            common_passages = find_matching_passages(
                suspicious_clean_text,
                source["text"],
                minimum_words=5
            )

            if common_passages:

                passage_matches_by_source.setdefault(
                    source["file_name"],
                    set()
                ).update(common_passages)

        if passage_matches_by_source:

            for source_name, passages in passage_matches_by_source.items():

                st.markdown(
                    f"**Source Document: {source_name}**"
                )

                for passage in sorted(passages):

                    st.success(
                        passage
                    )

        else:

            st.info(
                "No identical matching passages found."
            )

        st.caption(
            "Shared words and phrases are not automatically "
            "plagiarism. Common words and names may appear "
            "in unrelated documents."
        )

        # --------------------------------
        # TOP FIVE SEMANTIC MATCHES
        # --------------------------------

        st.header(
            "Top 5 Closest Semantic Matches"
        )

        if top_matches:

            top_data = []

            for number, result in enumerate(
                top_matches,
                start=1
            ):

                top_data.append({

                    "Rank": number,

                    "Source Document": get_source_name(result),

                    "Reference Type": result.get(
                        "source_type",
                        "PAN-PC-11 Dataset"
                    ),

                    "Similarity (%)": get_similarity_percent(
                        result["similarity_score"]
                    )

                })

            st.dataframe(
                pd.DataFrame(top_data),
                use_container_width=True
            )

        else:

            st.info(
                "No reference passages are available to compare."
            )

        # --------------------------------
        # PDF REPORT DOWNLOAD
        # --------------------------------

        st.header(
            "Download Plagiarism Report"
        )

        try:

            pdf_report = generate_pdf_report(
                document_name,
                matches
            )

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_report,
                file_name="plagiarism_report.pdf",
                mime="application/pdf",
                key="download_plagiarism_report"
            )

        except Exception as error:

            st.error(
                f"Unable to generate PDF report: {error}"
            )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Powered by Python, Streamlit, Sentence Transformers, "
    "FAISS, ReportLab and PAN-PC-11 Dataset."
)
