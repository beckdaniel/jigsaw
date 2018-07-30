import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.utils as U
import torch.optim as optim


def load_data(direc):
    with open('../data/num_letters/' + direc + '/input.txt') as f:
        X = [inst.split() for inst in f.readlines()]
    with open('../data/num_letters/' + direc + '/output_num.txt') as f:
        Y = [inst.split() for inst in f.readlines()]
    return X, Y


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        #if isinstance(sum_exp, Number):
        #    return m + math.log(sum_exp)
        #else:
        return m + torch.log(sum_exp)


class Net(nn.Module):
    

    def __init__(self):
        super(Net, self).__init__()
        self.vocab_size = 20
        self.dim = 200
        self.dim2 = 200
        self.length = 5
        self.embs = nn.Embedding(self.vocab_size, self.dim)
        self.lstm = nn.LSTM(self.dim, self.dim // 2, bidirectional=True)
        self.dec_lstm = nn.LSTM(self.dim, self.dim)
        self.in_layer = nn.Linear(self.dim, self.dim2)
        self.in_layer2 = nn.Linear(self.dim2, self.length)
        self.ones = torch.ones((self.length, self.length))
        self.out_layer = nn.Linear(self.dim, self.vocab_size)
        self.out_layer.weight = self.embs.weight
        self.hidden = self.init_hidden()
        self.dec_hidden = self.init_dec_hidden()
        self.cont_layer = nn.Linear(1, self.dim2)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.dim // 2),
                torch.zeros(2, 1, self.dim // 2))

    def init_dec_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.dim),
                torch.zeros(1, 1, self.dim))
        
    def forward(self, x):
        #print(x)
        #x = torch.tensor([int(i) for i in x])
        self.float_x = torch.tensor(x, dtype=torch.float).view(self.length, 1)
        #self.one_hot_x = self.make_one_hot(x)
        #print(self.float_x)
        embs_x = self.cont_layer(self.float_x)
        #print(embs_x)
        #embs_x = self.embs(x)
        lstm_x = embs_x
        #lstm_x, self.hidden = self.lstm(embs_x.view(self.length, 1, -1), self.hidden)
        #trans_x = self.in_layer(lstm_x.view(self.length, -1))
        #trans_x = F.tanh(trans_x)
        #trans_x = self.in_layer2(trans_x)
        #lstm_x = lstm_x.view(self.length, -1)
        trans_x = self.in_layer2(lstm_x)
        #trans_x = embs_x
        #print(x)
        #trans_x_t = torch.transpose(trans_x, 0, 1)
        #print(x_t)
        #print("TRANS_X")
        #print(trans_x)
        #sinkhorn = torch.matmul(trans_x, trans_x_t)
        sinkhorn = trans_x
        sinkhorn = sinkhorn / 0.01
        #sinkhorn = torch.exp(sinkhorn)
        #print(sinkhorn)
        for i in range(20):
            #print(sinkhorn)
            #col_norm = sinkhorn.matmul(self.ones)
            col_norm = logsumexp(sinkhorn, dim=0, keepdim=True)
            #sinkhorn = torch.div(sinkhorn, col_norm)
            sinkhorn = torch.sub(sinkhorn, col_norm)
            #print(sinkhorn)
            #row_norm = self.ones.matmul(sinkhorn)
            row_norm = logsumexp(sinkhorn, dim=1, keepdim=True)
            #sinkhorn = torch.div(sinkhorn, row_norm)
            sinkhorn = torch.sub(sinkhorn, row_norm)
            #print(sinkhorn)
        sinkhorn = torch.exp(sinkhorn)
        self.sh = sinkhorn
        #print(torch.sum(sinkhorn, dim=0))
        #self.reordered_x = torch.matmul(sinkhorn, embs_x)
        #print(reordered_x)
        # Not normalised outputs
        #not_norm = self.out_layer(self.reordered_x)
        #not_norm = F.tanh(not_norm)
        # Ouputs into prob dists
        #not_norm = self.reordered_x
        #self.outputs = F.log_softmax(not_norm)
        #self.outputs = torch.matmul(sinkhorn, self.float_x)
        #print(x)
        #print(sinkhorn)
        #print(self.outputs)
        self.reordered_embs_x = torch.matmul(sinkhorn, embs_x)
        #self.reordered_trans_x = torch.matmul(sinkhorn, lstm_x)
        #self.reordered_trans_x = lstm_x
        #self.reordered_trans_x = torch.transpose(self.reordered_trans_x, 0, 1)
        #print(self.reordered_trans_x)
        #print(self.one_hot_x)
        #reordered_x = torch.matmul(sinkhorn, self.one_hot_x)
        #print(reordered_x)
        self.outputs = torch.matmul(sinkhorn, self.float_x.view(self.length))
        #self.outputs = reordered_x
        #self.outputs = F.log_softmax(reordered_x)
        #self.dec_output, self.dec_hidden = self.dec_lstm(self.reordered_trans_x.view(self.length, 1, -1), self.dec_hidden)
        #print(self.outputs)
        #sys.exit(0)
        #self.outputs = torch.log(self.outputs + 1e-20)
        #print(self.outputs)
        #self.dec_output = self.out_layer(self.dec_output.view(self.length, -1))
        #self.dec_output = self.out_layer(self.reordered_trans_x)
        #self.outputs = F.log_softmax(self.dec_output)
        #print(self.outputs)
        #print(self.outputs)
        #sys.exit(0)
        return self.outputs

    def make_one_hot(self, x):
        one_hot = torch.zeros((self.length, self.vocab_size))
        for i, elem in enumerate(x):
            one_hot[i][elem] = 1
        return one_hot
        
    def reconstruction_loss(self, y):
        #embs_y = self.embs(y)
        float_y = torch.tensor(y, dtype=torch.float).view(5,1)
        embs_y = self.cont_layer(float_y)
        result = torch.norm(embs_y - self.reordered_embs_x)
        return result
        #float_y = torch.tensor(y, dtype=torch.float).view(5,1)
        #result2 = torch.sum(torch.pow(self.outputs - float_y, 2))
        #print(result)
        #print(embs_y)
        #print(self.reordered_x)
        #return result + result2
        #print(self.outputs)
        #print(float_y)
        #print(result2)
        #return result2
    


if __name__ == "__main__":
    X_train, Y_train = load_data('train')
    X_dev, Y_dev = load_data('dev')

    #print(X_train[:10])
    #print(Y_train[:10])

    #net = Net()
    #input_ = X_train[0]
    #target = Y_train[0]
    #output = net(input_)

    net = Net()
    #criterion = nn.CrossEntropyLoss(size_average=False)
    #criterion = nn.NLLLoss(size_average=False)
    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)#, momentum=0.9)


    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(zip(X_train, Y_train)):
            #print(data)
            x = torch.tensor([float(j) for j in data[0]]) / 10#, dtype=torch.float) 
            #y = torch.tensor([int(j) for j in data[1]])
            y = torch.tensor([float(j) for j in data[1]]) / 10
            #print(i)
            #print(x)
            #print(y)
            
            optimizer.zero_grad()
            net.hidden = net.init_hidden()
            net.dec_hidden = net.init_dec_hidden()

            output = net(x)
            #print(output)
            #break
            #print(output)
            #print(y)
            loss = criterion(output, torch.tensor(y, dtype=torch.float))
            #loss = criterion(output, y) + net.reconstruction_loss(y)
            #loss = net.reconstruction_loss(y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #print(i)
            #print(running_loss)
            if i % 200 == 199:    # print every 200 mini-batches
                exp_output = torch.exp(output)
                print(net.sh)
                #print(torch.argmax(exp_output, dim=1))
                print(output)
                print(y)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    for i, data in enumerate(zip(X_dev, Y_dev)):
        x = torch.tensor([int(j) for j in data[0]])
        y = torch.tensor([int(j) for j in data[1]])

        net.hidden = net.init_hidden()
        net.dec_hidden = net.init_dec_hidden()
        output = net(x)
        #_, predicted = torch.argmax(output, 1)
        predicted = torch.argmax(torch.exp(output), dim=1)
        #predicted = torch.tensor(output, dtype=torch.int)
        print("PREDICT:")
        print(predicted)
        print("TRUTH:")
        print(y)
        break
                
    #print(output)
    #net.zero_grad()
    #output.backward(torch.randn(10, 50))
    #net.forward(X_train[0])
    #print(list(net.parameters()))
