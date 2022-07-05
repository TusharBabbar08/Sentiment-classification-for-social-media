# %%
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
torch.cuda.empty_cache()

# %%
def read_file(file_list):
    '''
    Reads the txt file and assigns the parameters to respective list, updating the dictionary.
    Also, performing One Hot Encoding on the Sentiments.
    '''
    dataset = {}
    for path in file_list:
        dataset[path] = {}
        tweet = []
        tweetgts = []
        tweetid = []
        with open(path, encoding='utf8') as file:
            for line in file:
                line = line[:len(line) - 1]
                contents = line.split('\t')
                tweetid.append(int(contents[0]))
                if(contents[1] == 'positive'):
                    tweetgts.append([0, 1, 0])
                elif(contents[1] == 'negative'):
                    tweetgts.append([0, 0, 1])
                else:
                    tweetgts.append([1, 0, 0])
                tweet.append(contents[2])
        dataset[path]['tweet'] = tweet
        dataset[path]['sentiment'] = tweetgts
        dataset[path]['ids'] = tweetid
    return dataset
dataset = read_file(['twitter-training-data.txt', 'twitter-dev-data.txt','twitter-test1.txt','twitter-test2.txt','twitter-test3.txt'])

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%
def cleanup_text(texts):
    '''
    Pre-processed the tweets and returns a clean tweets after
    replacing and removing the unwanted bits and pieces from the tweet.
    '''
    cleaned_text = []
    for text in texts:
        # remove ugly &quot and &amp
        text = re.sub(r"&quot;(.*?)&quot;", "\g<1>", text)
        text = re.sub(r"&amp;", "", text)

        # replace emoticon
        text = re.sub(
            r"(^| )(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)",
            "\g<1>TOKEMOTICON",
            text,
        )

        text = text.lower()
        text = text.replace("tokemoticon", "TOKEMOTICON")

        # replace url
        text = re.sub(
            r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
            "TOKURL",
            text,
        )

        # replace mention
        text = re.sub(r"@[\w]+", "TOKMENTION", text)

        # replace hashtag
        text = re.sub(r"#[\w]+", "TOKTAG", text)

        # replace dollar
        text = re.sub(r"\£\d+", "TOKPOUND", text)

        # remove punctuation
        text = re.sub("[^a-zA-Z0-9]", " ", text)

        # remove multiple spaces
        text = re.sub(r" +", " ", text)

        # remove newline
        text = re.sub(r"\n", " ", text)
        
        #Remove Digits
        text= re.sub('[0-9\n]',' ',text)

        cleaned_text.append(text)
    return cleaned_text

# %%
cleaned_tweets = cleanup_text(dataset['twitter-training-data.txt']['tweet'])
tokenizer = Tokenizer(num_words = 5000, oov_token='<oov>')
tokenizer.fit_on_texts(cleaned_tweets)
word_index= tokenizer.word_index
train_tokenized_sentence = tokenizer.texts_to_sequences(cleaned_tweets)

# %%
def padding(seq, max_len = 45):
    '''
    Padding to make tweets same in length.
    Filling empty spaces with 0.
    '''
    pad_value = 0
    ls=[]
    for i in seq:
        pad_size = max_len - len(i)
        final_list = [*i, *[pad_value] * pad_size]
        ls.append(final_list)
    return ls
train_padded_seq = padding(train_tokenized_sentence)
# valid_padded_seq = padding(valid_tokenized_sentence)

# %%
glove = {}

with open('glove.6B.100d.txt', 'rb') as f:
    for l in f:
        line = str(l)
        line = line[2:len(line) - 3].split(' ')
        val = [float(line[i]) for i in range(1, len(line))]
        glove[line[0]] = val


matrix_len = len(word_index) + 1
weights_matrix = np.zeros((matrix_len, 100)) # 100 because this version of glove has 100 dims for a word
words_found = 0

for i, word in enumerate(word_index):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(100, ))
weights_matrix = torch.tensor(weights_matrix).to(device=device)

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embed, embed_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embed, embed_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embed, embed_dim

# %%
class LSTMClassifier(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, num_classes, dropout = 0.5):
        super(LSTMClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding, _, embedding_dim = create_emb_layer(weights_matrix, non_trainable=True)
        self.LSTM = nn.LSTM(embedding_dim, self.hidden_size, self.num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        embed = self.embedding(input)
        hidden = self.init_hidden(input.size(0))
        out, _ = self.LSTM(embed, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return F.softmax(out, dim=1)
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden
del weights_matrix



