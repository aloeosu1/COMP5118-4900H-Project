def amazon_knowledge(text: str) -> int:
    pos = ["amazing", "excellent", "love", "perfect", "awesome", "great", "highly recommend", "like", "good", "worth it","best", "nice"]
    neg = ["bad", "horrible", "terrible", "awful", "worst", "broke", "broken", "waste", "disappointed", "not good", "refund", "doesn't work", "didn't work"]

    t = text.lower()
    return sum(1 for k in (pos + neg) if k in t)


def agnews_knowledge(text: str) -> float:
    if not isinstance(text, str):
        return 0.0

    t = text.lower()

    keywords = ["president", "government", "election", "war", "conflict", "game", "match", "season", "league", "tournament", "market", "stock", "company", "profit", "economy", "technology", "tech", "software", "ai", "research"]

    score = 0
    for word in keywords:
        if word in t:
            score += 1

    return float(score)