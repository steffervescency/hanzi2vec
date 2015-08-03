class SegRNN(object):

    def __init__(self, hidden_dims, word2vec_dims, training_rate):
        self.hidden_dims = hidden_dims
        self.word2vec_dims = word2vec_dims
        self.training_rate = training_rate

        # TODO: Initialize these
        self.W = None
        self.Wscore = None
        self.b = None

    
    # Apply the activation function to a matrix
    # Socher's original paper used tanh, but the later recursive NN implementation used ReLU
    # so we should experiment with this
    def activation(matrix):
        # TODO: implement this
        return matrix

    # Apply one step of the forward-propogation to the two given nodes
    def forwardprop(node1, node2):
        p = self.activation(self.W*np.hstack(node1.val, node2.val) + self.b)
        # TODO: incorporate the lexical and structural information to the score
        score = self.Wscore * p
        return (score, p)
    
    # Apply the back-propogation algorithm to the whole parse tree
    def backprop(self, cost):
        # TODO: implementation
    
    # Parse a single sentence using the greedy algorithm
    def parse(self, sentence):
        # TODO: implement this
        # turn each character in the sentence into a node
        # while we don't have a completed tree
            # run forwardprop on all non-joined nodes
            # join the two nodes with the highest score
        # calculate the cost function
        return (tree, cost)
        
    # Train on the given data
    def train(data, num_iterations):
        # TODO: implementation
        
        # 1) Calculate the ML estimates of character sequences
        # 2) Create the negative training examples
        # 3) Split data into mini-batches
            # 4) Run forward propogation on each sentence, accumulating the cost
            # 5) Backpropogate