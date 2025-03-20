from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from ntlk_utils import bag_of_word, tokenize, correct_czech_text_api

app = Flask(__name__)

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Nerozumím. Můžete to zopakovat?"})

    user_input = correct_czech_text_api(user_input)  # Oprava překlepů
    sentence = tokenize(user_input)
    x = bag_of_word(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.85:
        for intent in trenovaci_data["intents"]:
            if tag == intent["tag"]:
                return jsonify({"response": random.choice(intent["responses"])})
    else:
        return jsonify({"response": "S tím by vám nejlépe pomohl můj lidský kolega."})


if __name__ == "__main__":
    app.run(debug=True)
