
import re
import difflib


# Common English words that should not be treated as meaningful matches
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "am", "be", "been", "being", "to", "of", "and",
    "or", "in", "on", "at", "for", "with", "from",
    "by", "as", "that", "this", "it", "i", "me",
    "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "their", "have", "has", "had",
    "do", "does", "did", "can", "could", "will",
    "would", "should", "no", "not", "so", "when",
    "more", "its", "own", "still", "often", "many",
    "what", "these", "those", "into", "because",
    "there", "which", "while", "also", "than",
    "then", "some", "such", "very", "after", "how",
    "keep", "let", "must", "need", "only", "who",
    "work", "using", "business", "company", "companies",
    "improve", "ready", "outside", "places", "members",
    "actions", "careful", "lower", "maintain", "manage",
    "although", "all", "about", "above", "across",
    "again", "against", "almost", "already", "always",
    "among", "any", "around", "before", "behind",
    "below", "between", "both", "but", "each",
    "either", "else", "ever", "every", "few", "first",
    "further", "here", "however", "just", "less",
    "much", "must", "never", "now", "off", "once",
    "other", "otherwise", "per", "same", "since",
    "than", "through", "throughout", "together",
    "under", "until", "upon", "where", "whether",
    "within", "without", "yet", "also", "able"
}


def tokenize(text):
    """Convert text into lowercase words."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def get_words(text):
    """Remove common words and very short words."""
    words = tokenize(text)

    return [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]


def find_exact_word_matches(suspicious_text, reference_text):
    """Find meaningful common words."""
    suspicious_words = set(get_words(suspicious_text))
    reference_words = set(get_words(reference_text))

    return sorted(
        suspicious_words.intersection(reference_words)
    )


def find_phrase_matches(
    suspicious_text,
    reference_text,
    phrase_length=4
):
    """Find identical phrases while ignoring common-only phrases."""

    suspicious_words = tokenize(suspicious_text)
    reference_words = tokenize(reference_text)

    def create_phrases(words):
        phrases = set()

        for i in range(len(words) - phrase_length + 1):
            phrase = words[i:i + phrase_length]

            meaningful_words = [
                word for word in phrase
                if word not in STOP_WORDS and len(word) > 2
            ]

            if len(meaningful_words) >= 3:
                phrases.add(tuple(phrase))

        return phrases

    suspicious_phrases = create_phrases(suspicious_words)
    reference_phrases = create_phrases(reference_words)

    common_phrases = suspicious_phrases.intersection(
        reference_phrases
    )

    return sorted(
        " ".join(phrase) for phrase in common_phrases
    )


def highlight_matching_words(suspicious_text, reference_text):
    """Return words that can be highlighted."""
    return find_exact_word_matches(
        suspicious_text,
        reference_text
    )


def highlight_matching_phrases(suspicious_text, reference_text):
    """Return phrases that can be highlighted."""
    return find_phrase_matches(
        suspicious_text,
        reference_text
    )


def find_matching_passages(
    suspicious_text,
    reference_text,
    minimum_words=5
):
    """Find matching passages with meaningful content."""

    suspicious_words = tokenize(suspicious_text)
    reference_words = tokenize(reference_text)

    matcher = difflib.SequenceMatcher(
        None,
        suspicious_words,
        reference_words,
        autojunk=False
    )

    matching_passages = []

    for block in matcher.get_matching_blocks():

        if block.size < minimum_words:
            continue

        passage_words = suspicious_words[
            block.a:block.a + block.size
        ]

        meaningful_words = [
            word for word in passage_words
            if word not in STOP_WORDS and len(word) > 2
        ]

        # Ignore passages with fewer than 3 meaningful words
        if len(meaningful_words) < 3:
            continue

        passage = " ".join(passage_words)

        if passage not in matching_passages:
            matching_passages.append(passage)

    return matching_passages
