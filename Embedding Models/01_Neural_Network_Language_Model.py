import torch
import torch.nn as nn


class NNLM(nn.Module):

    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(vocab_size, embedding_size)
        self.H = nn.Linear(num_steps * embedding_size, num_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(num_hidden))
        self.U = nn.Linear(num_hidden, vocab_size, bias=False)
        self.W = nn.Linear(num_steps * embedding_size, vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones(vocab_size))

    def forward(self, X):
        X = self.C(X) 
        X = X.view(-1, num_steps * embedding_size)
        tanh = torch.tanh(self.d + self.H(X))
        output = self.b + self.W(X) + self.U(tanh)
        return output


def batches():
    input_batch = []
    target_batch = []

    for sentence in sentences:
        word = sentence.split()
        input = [words_dict[n] for n in word[:-1]] # selecting n-1 words
        target = words_dict[word[-1]] # selecting last word

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

if __name__ == '__main__':
    # These variables are set based on paper
    num_steps = 2 # (n-1) steps
    num_hidden = 2 
    embedding_size = 2 
    
    # Data PreProcessing
    sentences = ["I like dog", "I love coffee", "I hate milk"]

    words = " ".join(sentences).split()
    words = list(set(words)) # create list of all unique words
    words_dict = {w : i for i, w in enumerate(words)}
    vocab_size = len(words_dict)
    words_num_dict = {i : w for i, w in enumerate(words)}
    
    # Model
    model = NNLM()

    # Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = batches()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch: {epoch+1}, cost = {loss:.5f}')
        loss.backward()
        optimizer.step()


# Prediction
predict = model(input_batch).data.max(1, keepdim=True)[1]
print([sentence.split()[:2] for sentence in sentences], ':', [words_num_dict[n.item()] for n in predict.squeeze()])