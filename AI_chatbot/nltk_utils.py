import numpy as np
import nltk #https://www.nltk.org/ <- give as posibility to make operation on text

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence): #split sentence into array of single words or symol
    return nltk.word_tokenize(sentence)


def stem(word): #find one part of the word that repeat in other word ex. ["organize", "organizes"] to ["organ", "organ"]
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words): #return the bag of word that mean function check is the tokem is in the same as the word, if it is give it 1 else 0 
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32) #implement 0 for each element in array
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1 #if the token is similar to word to the array is set 1
    return bag