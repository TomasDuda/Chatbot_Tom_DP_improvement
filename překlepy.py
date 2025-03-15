import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import requests
from spellchecker import SpellChecker
from Levenshtein import distance as levenshtein_distance


# Tokenizace
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Stemmatizace (manuální pro češtinu)
def stem_czech_word(word):
    word = word.lower()
    suffixes = ["ový", "ová", "ové", "ý", "á", "í", "ého", "ému", "em", "ech", "ích", "ou", "ům"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word


# Lemmatizace pomocí UDPipe API
def lemmatize_czech_word_api(word):
    url = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
    params = {
        "model": "czech-pdt-ud-2.15-241121",
        "data": word,
        "tokenizer": "presegmented",
        "tagger": "",
        "parser": ""
    }
    response = requests.post(url, data=params)
    if response.status_code == 200:
        result = response.json()
        processed_text = result.get("result", "")
        lines = processed_text.split("\n")
        for line in lines:
            if not line.startswith("#") and line.strip():
                parts = line.split("\t")
                if len(parts) > 2:
                    return parts[2]
    return word


# Korekce překlepů pomocí Levenshteinovy vzdálenosti
def correct_typos(word, vocabulary):
    if word in vocabulary:
        return word
    closest_match = min(vocabulary, key=lambda vocab_word: levenshtein_distance(word, vocab_word))
    if levenshtein_distance(word, closest_match) <= 2:  # Limit vzdálenosti na 2 pro podobnost
        return closest_match
    return word


# Korekce textu pomocí API Korektor
def correct_czech_text_api(text):
    url = "https://lindat.mff.cuni.cz/services/korektor/api/correct"
    params = {"data": text, "output": "text"}
    response = requests.post(url, data=params)
    if response.status_code == 200:
        return response.text.strip()
    return text


# Hlavní pipeline s opravou překlepů a zpracováním textu
def preprocess_text(sentence, vocabulary):
    # Korekce textu API
    corrected_text = correct_czech_text_api(sentence)

    # Tokenizace
    tokens = tokenize(corrected_text)

    # Korekce překlepů na úrovni tokenů
    corrected_tokens = [correct_typos(token, vocabulary) for token in tokens]

    # Lemmatizace
    lemmatized_tokens = [lemmatize_czech_word_api(token) for token in corrected_tokens]

    return lemmatized_tokens


# Příklad použití
vocabulary = ["kočka", "běžet", "překlep", "trvalo", "korekce"]  # Slovní zásoba
sentence = "Kočka běžeela přes překklep."  # Testovací věta s překlepy
processed_tokens = preprocess_text(sentence, vocabulary)
print("Zpracované tokeny:", processed_tokens)