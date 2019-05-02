"""
PLEASE DOCUMENT HERE

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math

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
    def __init__(self, structure):
        """Constructor - initializes dictionaries used to represent nn"""

        self._path_weights = {}
        self._node_weights = {}
        self._structure = structure
        self._max_layer = len(structure)

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
            past_total = structure[line]

    def initialize_random_weights(self):
        """Sets random starting weights"""

        for path in self._path_weights:
            self._path_weights[path] = random.random()

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
        start = 0
        end = self._structure[0]
        layer = 0
        while end <= node:
            layer += 1
            start = end
            end += self._structure[layer]
        return layer

    def back_propagation_learning(self, tests):
        self.initialize_random_weights()
        for pair in tests:
            self.forward_propagate(pair)
            self.back_propagate(pair)


    def inj(self, node):
        prev_layer = self.layer_of(node) - 1
        prev_nodes = self.get_layer_range(prev_layer)

        sum = 0
        for prev in range(prev_nodes[0], prev_nodes[1]):
            sum += self._path_weights[(prev, node)] * self._node_weights[prev]

        return sum

    def forward_propagate(self, pair):
        """helper function - Forward propogates a single pair"""
        first_layer = self.get_layer_range(0)[1]
        for node in range(first_layer):
            self._node_weights[node] = pair[0][node]

        for layer in range(1, self._max_layer):
            nodes = self.get_layer_range(layer)
            for j in range(nodes[0], nodes[1]):
                inj = self.inj(j)
                self._node_weights[j] = logistic(inj)

    def back_propagate(self, pair):
        delta = {}
        output_layer = self.get_layer_range(self._max_layer - 1)
        index = 0
        for node in range(output_layer[0], output_layer[1]):
            delta[node] = self.calculate_delta(node, delta, pair[1][0], True)
            index += 1
        for layer in range(self._max_layer - 2, 0, -1):
            nodes = self.get_layer_range(layer)
            for node in range(nodes[0], nodes[1]):
                delta[node] = self.calculate_delta(node, delta)

        for x,y in self._path_weights:
            self._path_weights[(x,y)] = self._path_weights[(x,y)] + .01 + self._node_weights[x] * delta[y]

    def calculate_delta(self, node, delta, y = None, output = False):
        print("Calculating delta of", node )
        aj = self._node_weights[node]
        if output:
            return (aj) * (1 - aj) * (y - aj)
        else:
            sum = 0
            j_layer = self.layer_of(node) + 1
            j_nodes = self.get_layer_range(j_layer)
            for j in range(j_nodes[0], j_nodes[1]):
                sum += self._path_weights[(node, j)] * delta[j]

            return logistic(aj) * (1 - logistic(aj)) * sum

def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    for example in training:
        print(example)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    nn = NeuralNetwork([3, 6, 3])
    nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
