# AI-Chatbot
# Coffee and Tea Shop Chatbot

## Overview
This project implements a simple chatbot using PyTorch, tailored for communication with customers in a coffee and tea shop. The bot is designed to answer various inquiries such as greetings, farewells, questions about store inventory, payment methods, delivery, store location, recommendations, and jokes.

The bot's knowledge base is stored in a JSON file called intents.json, containing responses to possible user queries. The bot is trained based on this data, enabling it to engage in meaningful conversations with users.

## Installation
Requirements
PyTorch: Installation instructions
NLTK: pip install nltk
If any errors occur, ensure to install nltk.tokenize.punkt:

```
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Running the Chatbot
After installing PyTorch, simply run: python chat.py
Training the Bot (Optional)
To train the bot from scratch, run: python training.py
After training, execute: python chat.py

## Contributing
This project was primarily developed for learning purposes in deep learning. Contributions are welcome, including adding new features, improving existing ones, or enhancing the training process.

## Disclaimer
While this chatbot serves as a useful learning tool, it may not exhibit advanced intelligence. However, it provides a practical demonstration of implementing a conversational agent using deep learning techniques.
