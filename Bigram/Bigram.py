import numpy as np
from BPE import BPETokenizer
from tqdm.autonotebook import tqdm
import re

class Bigram:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab

        self.vocab_size = len(self.vocab)
        # matrix cotaining the quantities of each token, note that it starts with ones to aply smoothing 
        self.bigram_matrix = np.ones((self.vocab_size, self.vocab_size))
        self.prob_matrix = np.zeros((self.vocab_size, self.vocab_size))
        # dictionary to optimize the enconding process
        self.encoded_dict = {}

    # mwthod that update the bigram matrix
    def update_bigram_matrix(self, text: str):
        # adds the init and end tokens to the text
        text = "<|s|> "+ text + " <|e|>"
        # split the text into works
        splitted_text = text.split(" ")
        tokens = []
        # for each word, checks if it was already encoded, if it was, repleca the word for its tokens,
        # if it wasnt, aply the tokenizer.encode funcion
        for word in splitted_text:
            val = self.encoded_dict.get(word, self.tokenizer.encode(word))
            tokens += val
            self.encoded_dict[word] = val
        
        # loop to update the bigram matrix
        for first_token, next_token in zip(tokens, tokens[1:]):
            self.bigram_matrix[first_token, next_token] +=1


    # method that set the probability matrix based on the bigram matrix
    def update_prob_matrix(self):
        self.prob_matrix = self.bigram_matrix / np.sum(self.bigram_matrix, axis=1, keepdims = True)
        
    # method to train the 
    def train(self, list_of_texts):
        for text in tqdm(list_of_texts):
            self.update_bigram_matrix(text)
            self.update_prob_matrix()

    # method to generate text based on the bigram prob matrix.
    def generate_text(self, num_tokens=20):
        tokens_list = [5004]
        for i in range(num_tokens):


            rng = np.random.default_rng()
            next_token = rng.multinomial(1, self.prob_matrix[tokens_list[-1]], size=1).argmax(axis=-1)[0]
            tokens_list.append(next_token)
        
        return self.tokenizer.decode(tokens_list)
