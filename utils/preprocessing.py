import pandas as pd
import re
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

replacement_dict = {
    # Slang
    "u": "you", "ur": "your", "r": "are", "ya": "you",
    "cuz": "because", "cus": "because", "coz": "because",
    "lol": "laughing out loud", "lmao": "laughing my ass off",
    "brb": "be right back", "idk": "i don't know",
    "omg": "oh my god", "btw": "by the way", "ttyl": "talk to you later",
    "smh": "shaking my head", "imo": "in my opinion", "fyi": "for your information",
    "yo": "hey", "nah": "no", "yea": "yeah", "holla": "holler",
    "shawty": "girl", "bae": "baby", "boo": "lover", "homie": "friend",
    "lit": "amazing", "dope": "cool", "ride": "car", "crib": "house",
    "thang": "thing", "cash": "money", "guap": "money",
    "flexin": "showing off", "ballin": "living large", "blessed": "fortunate",

    # Contractions
    "can't": "cannot", "won't": "will not", "don't": "do not", "didn't": "did not",
    "i'm": "i am", "it's": "it is", "that's": "that is", "there's": "there is",
    "isn't": "is not", "haven't": "have not", "hasn't": "has not", "wasn't": "was not",
    "weren't": "were not", "'cause": "because", "cause": "because",
    "she's": "she is", "he's": "he is", "who's": "who is", "what's": "what is",
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "ain't": "is not",
    "lemme": "let me", "gimme": "give me", "outta": "out of", "lotta": "lot of",
    "kinda": "kind of", "sorta": "sort of", "coulda": "could have",
    "woulda": "would have", "shoulda": "should have", "tryna": "trying to",
    "'twas": "it was", "now's": "now is", "i'll": "i will"
}

regex_patterns = [
    (re.compile(r"\b(\w+)'ll\b"), r"\1 will"),
    (re.compile(r"\b(\w+)'re\b"), r"\1 are"),
    (re.compile(r"\b(\w+)'ve\b"), r"\1 have"),
    (re.compile(r"\b(\w+)in'\b"), r"\1ing"),
]

combined_patterns = [
    # Word boundary: not preceded or followed by a word character
    (re.compile(r"(?<!\w)(" + "|".join(map(re.escape, replacement_dict.keys())) + r")(?!\w)"),
     lambda match: replacement_dict[match.group(0)])
] + regex_patterns

def normalize_text(text):
    for pattern, repl in combined_patterns:
        text = pattern.sub(repl, text)
    return text

def clean_lyrics(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = normalize_text(text)
    text = re.sub(r"[^a-zA-Z\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
