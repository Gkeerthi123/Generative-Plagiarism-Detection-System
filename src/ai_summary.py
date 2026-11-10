
import requests


def generate_ai_summary(
    document_name,
    highest_similarity,
    risk_level,
    matches
):
    similarity_percent = highest_similarity * 100

    # Generate a factual report when no matches cross the threshold
    if not matches:
        return f"""
# Plagiarism Analysis Report

**Document:** {document_name}

## 1. Overall Analysis

The document was analyzed using semantic similarity detection.

The highest similarity score among the closest results was
{similarity_percent:.2f}%.

## 2. Why Similarity Was Detected

Some content showed a degree of similarity with reference material.
However, no semantic matches crossed the selected similarity threshold.

## 3. Matched Reference Documents

No semantic matches were found above the selected threshold.

This does not mean that no reference documents were uploaded.
It means that no matching passages crossed the selected threshold.

## 4. Sections That Need Review

No specific matching sections were identified above the threshold.

## 5. Recommendations

- Review passages with relatively high similarity.
- Check whether sources are properly cited.
- Consider lowering the similarity threshold for further investigation.

## 6. Important Conclusion

The highest similarity score was {similarity_percent:.2f}%.
This score alone does not prove plagiarism.

Manual review is recommended before making any conclusion.
"""

    # Prepare actual match evidence for AI
    match_details = ""

    for number, match in enumerate(matches[:5], start=1):
        match_details += f"""
Match {number}
Reference document: {match.get('source_file', 'Unknown')}
Similarity: {match.get('similarity_score', 0) * 100:.2f}%

Uploaded passage:
{match.get('suspicious_text', '')[:500]}

Reference passage:
{match.get('matched_source_text', '')[:500]}

------------------------------
"""

    prompt = f"""
You are a careful plagiarism analysis assistant.

Analyze only the evidence provided below.

Document: {document_name}
Highest similarity: {similarity_percent:.2f}%
Risk indicator: {risk_level}

Actual matched passages:
{match_details}

Write a report with these headings:
1. Overall Analysis
2. Why Similarity Was Detected
3. Matched Reference Documents
4. Sections That Need Review
5. Recommendations
6. Important Conclusion

Rules:
- Mention only reference documents listed in the evidence.
- Never invent document names or passages.
- Never claim plagiarism is proven.
- Do not claim the entire document was copied.
- Explain that similarity is not proof of plagiarism.
- Keep the report simple and factual.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 600
                }
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json().get("response", "").strip()

        if not result:
            return "AI did not return a report. Please try again."

        return result

    except requests.exceptions.Timeout:
        return "AI report generation timed out. Please try again."

    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Please make sure Ollama is running."

    except Exception as error:
        return f"Unable to generate AI report: {error}"
