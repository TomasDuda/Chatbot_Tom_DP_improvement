import random
import json
import torch
from model import NeuralNet
from ntlk_utils import bag_of_word, tokenize, correct_czech_text_api
#from train import trenovaci_data, hidden_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('trenovaci_data.json', 'r', encoding='UTF-8') as f:
    trenovaci_data = json.load(f)

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
    if sentence == "exit":
        break
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

    if prob.item() > 0.75:
        for intent in trenovaci_data["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: S tím by vám nejlépe pomohl můj lidský kolega")