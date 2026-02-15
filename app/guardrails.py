BLOCKED_KEYWORDS = ["otp", "pin", "cvv", "card number"]

def is_safe(query: str) -> bool:
    query_lower = query.lower()
    return not any(word in query_lower for word in BLOCKED_KEYWORDS)
