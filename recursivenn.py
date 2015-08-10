import numpy as np
import pdb, sys
import random

class Node(object):
    
    def __init__(self, p, left=None, right=None, score=None):
        self.p = p
        self.left = left
        self.right = right
        self.score = score

class SegRNN(object):

    def __init__(self, word_vecs, word2vec_dim=200, training_rate=0.1,
                    error_margin=0.1, batch_size=100):
        self.word2vec_dim = word2vec_dim
        self.training_rate = training_rate
        self.error_margin = error_margin
        self.batch_size = batch_size
        
        self.word_vecs = word_vecs

        self.W = 0.01*np.random.randn(self.word2vec_dim, 2*self.word2vec_dim)
        self.Wscore = 0.01*np.random.randn(1, self.word2vec_dim)
        self.b = np.zeros((self.word2vec_dim))
        
        # TODO: Initialize these properly
        self.dW = None
        self.dWscore = None
        self.db = None
    
    # Apply the activation function
    # Socher's original paper used tanh, but the later recursive NN implementation used ReLU
    # so we should experiment with this
    def activation(self, inp):
        return np.tanh(inp)
    
    # The cost function - try to score the actual sentences higher than the negative examples
    def cost(self, sent_score, neg_sent_score):
        return max(0, self.error_margin - sent_score + neg_sent_score)
    
    # Score a node combination as a constituent
    # TODO: Add lexical and structural information
    #       and generally experiment with this function
    def score(self, p):
        return np.vdot(self.Wscore, p)

    # Apply one step of the forward-propogation to the two given nodes
    def forwardprop(self, node1, node2):
        p = self.activation(np.dot(self.W, np.hstack([node1.p, node2.p])) + self.b)
        score = self.score(p)
        return (score, p)
    
    # Apply the back-propogation algorithm to the given tree
    def backprop(self, node, cost):
        return True
        # TODO: calculate derivatives recursively
    
    
    # Parse a single sentence using the greedy algorithm
    def parse(self, sent):
        node_list = []
        score = 0
        for word in sent:
            node_list.append(Node(word))

        # Continue combining nodes as long as we don't have a full tree
        while len(node_list) > 1:
            max_score = float("-inf")
            max_idx = None
            max_p = 0
            for i in range(0, len(node_list) - 1):
                # Greedily search through each pair of nodes
                node1 = node_list[i]
                node2 = node_list[i+1]
                (score, p) = self.forwardprop(node1, node2)
                if score > max_score:
                    max_score = score
                    max_idx = i
                    max_p = p
            
            # Add a new node with the highest scoring pair
            node1 = node_list.pop(max_idx)
            node2 = node_list.pop(max_idx)
            combined = Node(max_p, node1, node2)
            node_list.insert(max_idx, combined)
            
            # Add to the score (just a sum of the local decisions)
            score += max_score
        
        return (node_list[0], score)
    
    # Produce a single negative example
    def negative_example(self, sent):
        idx = random.randint(0, len(sent) - 1)
        random_word = random.choice(self.word_vecs)
        neg_sent = list(sent)
        neg_sent[idx] = random_word
        return neg_sent
        
    # Train on the given data
    # TODO: Need to create the ML counts if we are going to use lexical information
    def train(self, data, num_iterations=5):
        counter = 0
        for it in range(0, num_iterations):
            print("Starting iteration #{0}\n".format(it+1))
            for idx, sent in enumerate(data):
                counter += 1
                if(len(sent) < 2):
                    continue
                
                # Create the negative example, score both
                neg_sent = self.negative_example(sent)
                (parsed, pos_score) = self.parse(sent)
                neg_score = self.parse(neg_sent)[1]
                cost = self.cost(pos_score, neg_score)
                
                # Calculate backprop derivatives
                self.backprop(cost, parsed)
                
                if counter == self.batch_size:
                    # TODO: update the weights
                    counter = 0
                    

        return True
    
    # Save the segmented output so we can compare it to the gold standard segmentation
    def save(self, file_name):
        # TODO: implement this
        return True
        

if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print("Usage: recursivenn.py word_vectors.txt input_data.txt segmented_output.txt")
    
    word_vec_path = sys.argv[1]
    input_text_path = sys.argv[2]
    save_path = sys.argv[3]
    
    word_vecs = {}
    sents = []
    
    with open(word_vec_path) as f:
        for line in f:
            cols = line.split()
            word_vecs[cols[0]] = np.array([float(val) for val in cols[1:]])
    
    with open(input_text_path) as f:
        for line in f:
            words = [word_vecs.get(word, np.zeros(200)) for word in line.split()]
            sents.append(words)
    
    nn = SegRNN(list(word_vecs.values()))
    nn.train(sents)
    nn.save(save_path)