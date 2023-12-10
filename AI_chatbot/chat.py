import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch is checking does our pc have nvidia cuda on gpu (not every graphics card have it) otherwise use cpu

with open('intents.json', 'r') as json_data: #import data from konwlagebase
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE) #import learning data set

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
print("Let's chat! (type 'quit' to exit, use english - the bot is giving answer about e-commerce shop with caffee and tea) ")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1) #softmax change probability to number beetwen range [0,1]
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}") #if bot found correct answer give on output random answer
    else:
        print(f"{bot_name}: I do not understand, ask me somethng else!")