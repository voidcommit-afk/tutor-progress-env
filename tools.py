def extract_concepts(chat_history):
    text = " ".join(chat_history).lower()

    concepts = []
    keywords = ["photosynthesis", "chlorophyll", "force", "numericals", "algebra", "fractions"]

    for k in keywords:
        if k in text:
            concepts.append(k)

    return concepts


def detect_weakness(chat_history):
    text = " ".join(chat_history).lower()

    if "numerical" in text:
        return "numerical problem solving"
    if "forget" in text:
        return "memory retention"
    if "panic" in text:
        return "exam anxiety"

    return "general conceptual gap"