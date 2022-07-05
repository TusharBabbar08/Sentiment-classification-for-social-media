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
weights_matrix = torch.tensor(weights_matrix)

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embed, embed_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embed, embed_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embed, embed_dim

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