"""
Mitchel Herman and Judy Zhou

Implementation of the Neural Network algorithm given to us in class
as well as some code used to run tests

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math, statistics, copy

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here


class NeuralNetwork():
    def __init__(self, structure, epochs):
        """Constructor - initializes dictionaries used to represent nn"""

        #Dictionary takes tuple of two nodes and returns the path weight between them
        self._path_weights = {}
        #Dictionary takes a node and returns its value
        self._node_weights = {}
        #keep track of the strucutre
        self._structure = structure
        self._max_layer = len(structure) - 1
        self._epochs = epochs

        #initialize node_weights dictionary used for propogation all set to None
        node_number = 0
        for line in structure:
            for node in range(line):
                self._node_weights[node_number] = None
                node_number += 1

        #initialize _path_weights dictionary - main implementation of NN
        past_total = 0
        total = structure[0]
        for line in range(len(structure) - 1):
            for node in range(structure[line]):
                for next in range(structure[line + 1]):
                    self._path_weights[(node + past_total, next + total)] = None

            total += structure[line + 1]
            past_total += structure[line]

        #Don't forget the dummy weights
        for node in self._node_weights:
            self._path_weights[("d",node)] = None

        self._node_weights["d"] = 1.0



    def initialize_random_weights(self):
        """Sets random starting weights"""

        for path in self._path_weights:
            self._path_weights[path] = random.uniform(-1,1)

    def get_layer_range(self, i):
        """returns a range of node numbers for a given layer"""

        start = 0
        end = self._structure[0]
        layer = 0
        while layer < i:
            layer += 1
            start = end
            end += self._structure[layer]

        return (start, end)

    def layer_of(self, node):
        """Given a node number, it returns the layer it is on"""
        start = 0
        end = self._structure[0]
        layer = 0
        while end <= node:
            layer += 1
            start = end
            end += self._structure[layer]
        return layer

    def clear_node_weights(self):
        """Resets all the values of the node before the next propogation"""
        for node in self._node_weights:
            self._node_weights[node] = None
        self._node_weights["d"] = 1.0

    def back_propagation_learning(self, tests):
        """Main Training code"""
        #print(self._path_weights)
        self.initialize_random_weights()
        for epoch in range(self._epochs):
            #print("Epoch : ", epoch)
            for pair in tests:
                self.forward_propagate(pair[0])
                self.back_propagate(pair)
                self.clear_node_weights()

    def inj(self, node):
        """Helper function to calcualte inj of a node"""
        prev_layer = self.layer_of(node) - 1
        prev_nodes = self.get_layer_range(prev_layer)

        sum = 0
        for prev in range(prev_nodes[0], prev_nodes[1]):
            sum += self._path_weights[(prev, node)] * self._node_weights[prev]
        #Don't forget the dummy node
        sum += self._path_weights[("d", node)] * self._node_weights["d"]

        return sum

    def guess(self, inputs):
        """Predict output of an input using the neural net"""
        self.forward_propagate(inputs[0])
        output_layer = self.get_layer_range(self._max_layer)
        res = []
        for node in range(output_layer[0], output_layer[1]):
            res.append(self._node_weights[node])

        return res

    def forward_propagate(self, inputs):
        """helper function - Forward propogates a single pair"""
        first_layer = self.get_layer_range(0)[1]
        for node in range(first_layer):
            self._node_weights[node] = inputs[node]

        for layer in range(1, self._max_layer + 1):
            nodes = self.get_layer_range(layer)
            for j in range(nodes[0], nodes[1]):
                inj = self.inj(j)
                self._node_weights[j] = logistic(inj)

    def back_propagate(self, pair):
        """Helper method - backward propagates a single pair"""
        delta = {}
        output_layer = self.get_layer_range(self._max_layer)
        index = 0
        for node in range(output_layer[0], output_layer[1]):
            delta[node] = self.calculate_delta(node, delta, pair[1][index], True)
            index += 1
        for layer in range(self._max_layer - 1, -1, -1):
            nodes = self.get_layer_range(layer)
            for node in range(nodes[0], nodes[1]):
                delta[node] = self.calculate_delta(node, delta)

        for x,y in self._path_weights:
            self._path_weights[(x,y)] = self._path_weights[(x,y)] + (.1 * (self._node_weights[x] * delta[y]))

    def calculate_delta(self, node, delta, y = None, output = False):
        """Helper function calculates delta"""
        aj = self._node_weights[node]
        if output:
            return (aj) * (1 - aj) * (y - aj)
        else:
            sum = 0
            j_layer = self.layer_of(node) + 1
            j_nodes = self.get_layer_range(j_layer)
            for j in range(j_nodes[0], j_nodes[1]):
                sum += self._path_weights[(node, j)] * delta[j]

            return aj * (1 - aj) * sum

def simple_accuracy(nn, examples):
    """Calculates accuracy"""
    good = 0
    bad = 0
    for example in examples:
        done = False
        prediction = nn.guess(example)
        for i in range(len(example[1])):
            if not done:
                if example[1][i] != round(prediction[i]):
                    bad += 1
                    done = True
            if not done:
                good += 1

    return good/(good + bad)

def k_fold_cross_validation(k, data, nn):
    """For getting accuracy of the Neural network."""
    #Create the buckets
    buckets = [[] for _ in range(k)]
    bucket_size = len(data) // k

    #Populate each bucket randomly
    for bucket in buckets:
        for _ in range(bucket_size):
            pair = random.choice(data)
            data.remove(pair)
            bucket.append(pair)

    #Run the NN k times, with one bucket left out as testing.
    accuracy = []
    for i in range(k):
        training = [pair for bucket in buckets[:i] + buckets[i+1:] for pair in bucket]
        eval = buckets[i]
        nn.back_propagation_learning(training)
        accuracy.append(simple_accuracy(nn,eval))

    #Report average accuracy over k runs
    return(statistics.mean(accuracy))


def test_layers(data):
    """Ignore this - Just tesing code"""
    test_data = copy.deepcopy(data)
    nn1 = NeuralNetwork([13,10,3], 1000)
    #nn5 = NeuralNetwork([2,3,3,1], 1000)
    #nn10 = NeuralNetwork([2,3,3,3,1], 1000)
    #nn15 = NeuralNetwork([2,3,3,3,3,1], 1000)
    #nn20 = NeuralNetwork([2,3,3,3,3,3,1], 1000)

    print("TEST 1")
    accuracy1 = k_fold_cross_validation(5, test_data, nn1)
    print("Accuracy 1:",accuracy1)

    #print("TEST 5")
    #test_data = copy.deepcopy(data)
    #accuracy5 = k_fold_cross_validation(5, test_data, nn5)
    #print("Accuracy 5:",accuracy5)

    #print("TEST 10")
    #test_data = copy.deepcopy(data)
    #accuracy10 = k_fold_cross_validation(5, test_data, nn10)
    #print("Accuracy 10:",accuracy10)

    #print("TEST 15")
    #test_data = copy.deepcopy(data)
    #accuracy15 = k_fold_cross_validation(5, test_data, nn15)

    #print("TEST 20")
    #test_data = copy.deepcopy(data)
    #accuracy20 = k_fold_cross_validation(5, test_data, nn20)

    print("Accuracy 1:",accuracy1)
    #print("Accuracy 5:",accuracy5)
    #print("Accuracy 10:",accuracy10)
    #print("Accuracy 15:",accuracy15)
    #print("Accuracy 20:",accuracy20)


def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]
    print(training[0])
    real_training = [training[i] for i in range(0,len(training),2)]
    eval = [training[i+1] for i in range(0,len(training)-1, 2)]

    print(training)
    # Check out the data:
    for example in training:
        print(example)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    #nn = NeuralNetwork([13, 6, 3], 10000)
    #nn.back_propagation_learning(real_training)

    #for example in eval:
    #    guess = nn.guess(example)
    #    rounded = [round(item) for item in guess]
    #    print(example[1], ":", rounded)

    #simple_accuracy(nn, eval)

    #k_fold_cross_validation(5, training, nn)

    test_layers(training)

if __name__ == "__main__":
    main()
