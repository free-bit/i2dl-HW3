import torch
import torch.nn as nn

act_layer = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh
}

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # DONE: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialize h as 0 if these values are not given.                    #
        #######################################################################
        self.hidden_size = hidden_size
        self.ifc = nn.Linear(input_size, hidden_size)
        self.hfc = nn.Linear(hidden_size, hidden_size)
        self.active = act_layer[activation]()
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_seq = torch.Tensor().to(device)
        seq_len, batch_size, input_size = x.shape
        hidden_size = self.hidden_size

        # Initialize if not given
        if not h:
            h = torch.zeros(1, batch_size, hidden_size).to(device)
        
        for i in range(seq_len):
            h_prev = self.hfc(h)                     # Take h_prev (h of previous layer) and feed it into the layer
            h_now = self.ifc(x[i].unsqueeze_(0))     # Calculate h_now for the current input
            h = self.active(h_prev + h_now)          # Calculate h_next
            h_seq = torch.cat((h_seq, h), dim=0)     # Store it in a tensor (vstack)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # DONE: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################
        self.hidden_size = hidden_size

        self.hffc = nn.Linear(hidden_size, hidden_size) # hidden forget-gate FC (see f_t formula)
        self.iffc = nn.Linear(input_size, hidden_size)  # input forget-gate FC (see f_t formula)
        
        self.hifc = nn.Linear(hidden_size, hidden_size) # hidden input-gate FC (see i_t formula)
        self.iifc = nn.Linear(input_size, hidden_size)  # input input-gate FC (see i_t formula)
        
        self.hofc = nn.Linear(hidden_size, hidden_size) # hidden output-gate FC (see o_t formula)
        self.iofc = nn.Linear(input_size, hidden_size)  # input output-gate FC (see o_t formula)
        
        self.hcfc = nn.Linear(hidden_size, hidden_size) # (see c_t formula)
        self.icfc = nn.Linear(input_size, hidden_size)  # (see c_t formula)
        
        self.gsig = nn.Sigmoid()
        self.csig = nn.Tanh()
        self.hsig = nn.Tanh()
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = None
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_seq = torch.Tensor().to(device)
        seq_len, batch_size, input_size = x.shape
        hidden_size = self.hidden_size

        # Initialize if not given
        if not h:
            h = torch.zeros(1, batch_size, hidden_size).to(device)
        if not c:
            c = torch.zeros(1, batch_size, hidden_size).to(device)

        for i in range(seq_len):
            c_prev = c
            inp = x[i].unsqueeze_(0) # Add extra dimension to the beginning

            # f_t
            fh_prev = self.hffc(h)
            fi = self.iffc(inp)
            f_now = self.gsig(fi + fh_prev)

            # i_t
            ih_prev = self.hifc(h)
            ii = self.iifc(inp)
            i_now = self.gsig(ii + ih_prev)

            # o_t
            oh_prev = self.hofc(h)
            oi = self.iofc(inp)
            o_now = self.gsig(oi + oh_prev)

            # c_t
            ch_prev = self.hcfc(h)
            ci = self.icfc(inp)
            c = f_now * c_prev + i_now * self.csig(ci + ch_prev)

            # h_t
            h = o_now * self.hsig(c)
            h_seq = torch.cat((h_seq, h), dim=0) # vstack

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  DONE: Build a RNN classifier                                       #
        #######################################################################
        self.rnn = RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, classes)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        _, x = self.rnn(x)
        x = self.fc(x)
        x.squeeze_(0)
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  DONE: Build a LSTM classifier                                      #
        #######################################################################
        self.lstm = LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, classes)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        x = self.fc(x)
        x.squeeze_(0)
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
