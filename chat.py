import difflib
import random
import os
import json
import torch
from model import NeuralNet
from ntlk_utils import bag_of_word, tokenize, correct_czech_text_api, lemmatize_czech_word_api

#from train import trenovaci_data, hidden_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('trenovaci_data.json', 'r', encoding='UTF-8') as f:
    trenovaci_data = json.load(f)

# Soubor pro učení nových vět
NOVE_VETY_FILE = "nove_vety.json"

if os.path.exists(NOVE_VETY_FILE):
    with open(NOVE_VETY_FILE, 'r', encoding='UTF-8') as f:
        nove_vety = json.load(f)
else:
    nove_vety = {"intents": []}

# Načtení modelu
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Tom"
print("S čím Vám mohu pomoci?")
while True:
    sentence = input('You:')

    # Nejprve zkontrolovat, zda je zadán příkaz pro ukončení
    if sentence.strip().lower() == "exit":
        print(f"{bot_name}: Děkuji za rozhovor, mějte se hezky!")
        break

    # Pokračovat ve zpracování věty, pokud není exit
    #sentence = lemmatize_czech_word_api(sentence)
    sentence = correct_czech_text_api(sentence)
    sentence = tokenize(sentence)
    x = bag_of_word(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.90:
        for intent in trenovaci_data["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: S tím by vám nejlépe pomohl můj lidský kolega")

        naucit_se = input("Mám se naučit tuto větu? (ano/ne): ").strip().lower()
        if naucit_se == "ano":
            nova_odpoved = input("Jaká by měla být správná odpověď? ")

            # Automatický návrh tagu podle podobnosti s existujícími
            mozny_tag = difflib.get_close_matches(sentence, tags, n=1, cutoff=0.5)
            if mozny_tag:
                navrzeny_tag = mozny_tag[0]
                print(f"Navržený tag: {navrzeny_tag}")
                potvrdit_tag = input(f"Použít tento tag? (ano/ne): ").strip().lower()
                if potvrdit_tag == "ne":
                    navrzeny_tag = input(f"Zadejte nový tag: ")
            else:
                navrzeny_tag = input(f"Zadejte nový tag: ")

            # Uložení nové věty do souboru
            nove_vety["intents"].append({
                "tag": navrzeny_tag,
                "pattern": [" ".join(sentence)],
                "responses": [nova_odpoved]
            })

            with open(NOVE_VETY_FILE, 'w', encoding='UTF-8') as f:
                json.dump(nove_vety, f, indent=4, ensure_ascii=False)

            print("Děkuji! Naučil jsem se novou odpověď.")