import numpy as np
from BPE import BPETokenizer
from tqdm.autonotebook import tqdm
import math

# this class is basicaly the same as the Bigram class, however, instead of predicting the next word, it computes the perplexity
class Perplexity:

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

    # method to compute perplexity
    def compute_perplexity(self, text: str):
        tokens = self.tokenizer.encode(text)
        perplexity =1
        count = 0
        for first_token, next_token in zip(tokens, tokens[1:]):
            perplexity = perplexity * self.prob_matrix[first_token, next_token]
            count +=1
        
        return math.pow(perplexity, -1/count)

