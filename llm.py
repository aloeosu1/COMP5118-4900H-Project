from openai import OpenAI

client = OpenAI()  

#amazon llm query
def llm_label(review_text: str) -> int:
    prompt = f"""
You are labeling Amazon product reviews for sentiment analysis.

Label this review as:
1 if the overall sentiment is clearly positive.
0 if the overall sentiment is clearly negative.

If the review seems mixed or neutral, choose the label that best matches the dominant tone.

Only respond with a single digit: 0 or 1.

Review:
\"\"\"{review_text}\"\"\"
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0.0,
    )

    text = resp.choices[0].message.content.strip()

    #look for 0 or 1
    if "1" in text and "0" not in text:
        return 1
    if "0" in text and "1" not in text:
        return 0

    #return 0 last resort
    return 0


#ag newss llm query
def llm_label_agnews(text: str) -> int:
    prompt = f"""
Classify this news text into one category. Only return a single digit 0-3.

0 = World (international, politics, government, conflicts)
1 = Sports (teams, games, athletes, leagues)
2 = Business (markets, economy, companies, finance)
3 = Sci/Tech (technology, science, computing, gadgets, research)

Text:
\"\"\"{text}\"\"\"
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=1,
    )
    out = (resp.choices[0].message.content or "").strip()
    if out in {"0","1","2","3"}:
        return int(out)
    #return 0 last resort
    return 0