import nltk
import numpy as np
#nltk.download('punkt_tab')

#from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import requests
from spellchecker import SpellChecker




def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#Tokenizace test
#a = "Jak dlouho to trvá?"
#print(a)
#a = tokenize(a)
#print(a)

def stem_czech_word(word):
    word = word.lower()
    suffixes = ["ový", "ová", "ové", "ý", "á", "í", "ého", "ému", "em", "ech", "ích", "ou", "ům"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

#Stemming test
#words = ["Trvání", "tRvalo", "trvat"]
#stemmed_words = [stem_czech_word(word) for word in (words)]
#print(stemmed_words)



def bag_of_word(tokenized_sentence, all_words):
    tokenized_sentence = [stem_czech_word(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


def lemmatize_czech_word_api(word):
    """Pošle slovo na UDPipe API a vrátí jeho lemma."""
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

        # Najdeme lemma ve výstupu
        lines = processed_text.split("\n")
        for line in lines:
            if not line.startswith("#") and line.strip():
                parts = line.split("\t")
                if len(parts) > 2:
                    return parts[2]  # Vrací lemma slova

    return word  # Pokud API selže, vrátí původní slovo


# Test lemmatizace přes API
#print(lemmatize_czech_word_api("překlep"))  # Výstup: "kočka"
#print(lemmatize_czech_word_api("běžela"))  # Výstup: "běžet"



def correct_czech_text_api(text):
    """Pošle text na Korektor API a vrátí opravenou verzi."""
    url = "https://lindat.mff.cuni.cz/services/korektor/api/correct"
    params = {"data": text, "output": "text"}

    response = requests.post(url, data=params)

    if response.status_code == 200:
        return response.text.strip()  # Korektor vrací opravený text přímo

    return text  # Pokud API selže, vrátí původní text


# Test korekce přes API
#print(correct_czech_text_api("Překklep."))  # Výstup: "Toto je špatně napsaná věta."









