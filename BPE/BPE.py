import regex as re
import tqdm
import os
import json

class BPETokenizer:

    def __init__(self) -> None:
        self.vocab = {index: bytes([index]) for index in range(256)}
        self.tokenizer_merges = {}
        pass

    def build_tokens_list(self, text: str):
        # encode the string o to utf-8 and set it as a list
        return list(text.encode("utf-8"))

    # method to count the consecutive tokens in a string and return the dictionary of its occurencies.
    def count_consecutive_tokens(self, tokens_list: list):
        # initiate the dict of pairs
        pair_counts ={}
        # compute all pair of tokens
        pair_of_tokens = zip(tokens_list, tokens_list[1:])
        # for each pair, add one to the pair_couts dictionary key
        for pair in pair_of_tokens:
            # get(pair, 0) sets the value to 0 if the key does not exist
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pair_counts
    
    # merges the token pair to a single new token and return a new token list
    def merge_tokens(self, tokens_list: list, pair: tuple, pair_token_id: int):
        # initiate the new list of tokens
        new_token_list = []
        i = 0
        while i < len(tokens_list):
            # add the pair as a new token
            if i < len(tokens_list)-1 and tokens_list[i] == pair[0] and tokens_list[i+1] == pair[1]:
                new_token_list.append(pair_token_id)
                # since we are merging the values, we need to jump the next token, thus, we add 2
                i+= 2
            # add the previuos token to the token list
            else:
                new_token_list.append(tokens_list[i])
                i+=1

        return new_token_list
    
    # return the most frequent pair in the pair list
    def get_most_frequent_pair(self, pair_counts: dict):
        return max(pair_counts, key=pair_counts.get)

    # add new token to vocabulary
    def add_pair_to_vocab(self, pair: tuple, pair_token_id: int):
        #working with bytes, the vocab will add a concatenations of the bytes
        self.vocab[pair_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]] 

    # add the merge to the merges list
    def add_merge_to_merges_list(self, merged_pair, pair_token_id):
        self.tokenizer_merges[merged_pair] = pair_token_id
        
    # reset vocab to the initial value
    def reset_vocab(self):
        self.vocab = {index: bytes([index]) for index in range(256)}
    
    # reset merges to the initial value
    def reset_tokenizer_merges(self):
        self.tokenizer_merges = {}
    
    # apply regex to text
    def apply_regex(self, text: str):
        # this regex was based on GPT-2 regex, however, I decided only to split the words, and numbers, and pontuations
        regex_rule = re.compile(r"""'s| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        splitted_text = re.findall(regex_rule, text)

        return splitted_text

    def get_files_list(self, directory: str):
        return os.listdir(directory)


    def tokenizer_fit_multidocs(self, desireble_size: int, directory: str):
        # reset the vocab and merges_list of the tokenizer
        self.reset_vocab()
        self.reset_tokenizer_merges()
        # compute que number of iterations
        vocab_len = len(self.vocab)
        number_iterations = desireble_size - vocab_len

        # read all files into a doc
        files = self.get_files_list(directory)
        files_text = []
        for file in tqdm.tqdm(files):
            file = os.path.join(directory, file)
            with open(file, 'r') as f:
                data = json.load(f)
                files_text .append(data['text'])



        files_text_word_list = [self.apply_regex(text) for text in files_text]

        
        # build the first set of tokens
        # tokens = self.build_tokens_list(text)
        tokens_list = []
        for word_list in tqdm.tqdm(files_text_word_list):
            files_tokens_list = [self.build_tokens_list(word) for word in word_list]
            tokens_list += files_tokens_list


        for i in tqdm.tqdm(range(number_iterations)):
            pair_token_id = vocab_len + i
            pair_counts = {}
            for tokens in tokens_list:
                for pair, quantity in self.count_consecutive_tokens(tokens).items():
                    pair_counts[pair] = pair_counts.get(pair, 0) + quantity
            
            most_frequent_pair = self.get_most_frequent_pair(pair_counts)

            for index, tokens in enumerate(tokens_list):
                tokens_list[index] = self.merge_tokens(tokens, most_frequent_pair, pair_token_id)

            
            # self.tokenizer_merges[most_frequent_pair] = pair_token_id
            # add the merge and the vocab to the tokenizer variables
            self.add_merge_to_merges_list(most_frequent_pair, pair_token_id)
            self.add_pair_to_vocab(most_frequent_pair, pair_token_id)
            #check if the there is still tokens to be merged
            if max([len(tokens) for tokens in tokens_list]) == 1:
                print("No more tokens to merge")
                print(f'only performed {i+1} iteretions of {number_iterations}')
                break

    def tokenizer_fit(self, desireble_size: int, text: str):
        # reset the vocab and merges_list of the tokenizer
        self.reset_vocab()
        self.reset_tokenizer_merges()
        # compute que number of iterations
        vocab_len = len(self.vocab)
        number_iterations = desireble_size - vocab_len

        # split text based on regex
        word_list = self.apply_regex(text)

        
        # build the first set of tokens
        # tokens = self.build_tokens_list(text)
        
        tokens_list = [self.build_tokens_list(word) for word in word_list]


        for i in tqdm.tqdm(range(number_iterations)):
            pair_token_id = vocab_len + i
            pair_counts = {}
            for tokens in tokens_list:
                for pair, quantity in self.count_consecutive_tokens(tokens).items():
                    pair_counts[pair] = pair_counts.get(pair, 0) + quantity
            
            most_frequent_pair = self.get_most_frequent_pair(pair_counts)

            for index, tokens in enumerate(tokens_list):
                tokens_list[index] = self.merge_tokens(tokens, most_frequent_pair, pair_token_id)

            
            # self.tokenizer_merges[most_frequent_pair] = pair_token_id
            # add the merge and the vocab to the tokenizer variables
            self.add_merge_to_merges_list(most_frequent_pair, pair_token_id)
            self.add_pair_to_vocab(most_frequent_pair, pair_token_id)
            #check if the there is still tokens to be merged
            if max([len(tokens) for tokens in tokens_list]) == 1:
                print("No more tokens to merge")
                print(f'only performed {i+1} iteretions of {number_iterations}')
                break


    def decode(self, tokens_list: list):

        tokens = b"".join(self.vocab[index] for index in tokens_list)
        decoded_text = tokens.decode('utf-8', errors="replace")

        return decoded_text

    def encode(self, text: str):
        tokens_list = self.build_tokens_list(text)

        while(len(tokens_list) >= 2):
            count_pairs = self.count_consecutive_tokens(tokens_list)
            # looks for the pairs in counting pairs that appears in the merge_tokens variable
            # it will the pair of tokens with smallest index in merged tokens
            # this is necessary to merge the tokens in the order they were added in the merged_tokens variable.
            pair = min(count_pairs, key=lambda pair: self.tokenizer_merges.get(pair, float("inf")))
            if pair not in self.tokenizer_merges:
                break
            pair_token_id = self.tokenizer_merges[pair]
            tokens_list = self.merge_tokens(tokens_list, pair, pair_token_id)
        return tokens_list









