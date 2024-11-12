import numpy as np
from BPE import BPETokenizer
from tqdm.autonotebook import tqdm
import math

class Perplexity:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        # build add and and 

        # build matrix of occurency
        self.vocab_size = len(self.vocab)
        self.bigram_matrix = np.ones((self.vocab_size, self.vocab_size))
        self.prob_matrix = np.zeros((self.vocab_size, self.vocab_size))
        self.encoded_dict = {}


    def update_bigram_matrix(self, text: str):
        text = "<|s|> "+ text + " <|e|>"
        splitted_text = text.split(" ")
        tokens = []
        for word in splitted_text:
            val = self.encoded_dict.get(word, self.tokenizer.encode(word))
            tokens += val
            self.encoded_dict[word] = val
        # tokens = self.tokenizer.encode(text)
        # print(tokens)
        for first_token, next_token in zip(tokens, tokens[1:]):
            self.bigram_matrix[first_token, next_token] +=1


    
    def update_prob_matrix(self):
        self.prob_matrix = self.bigram_matrix / np.sum(self.bigram_matrix, axis=1, keepdims = True)
        

    def train(self, list_of_texts):
        for text in tqdm(list_of_texts):
            self.update_bigram_matrix(text)
            self.update_prob_matrix()

    
    def compute_perplexity(self, text: str):
        tokens = self.tokenizer.encode(text)
        perplexity =1
        count = 0
        for first_token, next_token in zip(tokens, tokens[1:]):
            perplexity = perplexity * self.prob_matrix[first_token, next_token]
            count +=1
        
        return math.pow(perplexity, -1/count)
        # tokens_list = [5004]
        
    # def compute_perplexity(self, num_tokens=20):
    #     tokens_list = [5004]
    #     for i in range(num_tokens):


    #         rng = np.random.default_rng()
    #         next_token = rng.multinomial(1, self.prob_matrix[tokens_list[-1]], size=1).argmax(axis=-1)[0]
    #         tokens_list.append(next_token)
        
    #     return self.tokenizer.decode(tokens_list)
