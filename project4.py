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
    def __init__(self, structure, epochs):
        """Constructor - initializes dictionaries used to represent nn"""

        self._path_weights = {}
        self._node_weights = {}
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
            past_total = structure[line]

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
        start = 0
        end = self._structure[0]
        layer = 0
        while end <= node:
            layer += 1
            start = end
            end += self._structure[layer]
        return layer

    def clear_node_weights(self):
        for node in self._node_weights:
            self._node_weights[node] = None
        self._node_weights["d"] = 1.0

    def back_propagation_learning(self, tests):
        self.initialize_random_weights()
        #print("Initial Weights: ", self._path_weights)
        for epoch in range(self._epochs):
            print("Epoch : ", epoch)
            for pair in tests:
                self.forward_propagate(pair[0])
                self.back_propagate(pair)
                self.clear_node_weights()
        #print()
        #print()
        #print("End weights: ", self._path_weights)


    def inj(self, node):
        #print("INJ of Node:", node)
        prev_layer = self.layer_of(node) - 1
        prev_nodes = self.get_layer_range(prev_layer)
        #print("ON LAYER: ", self.layer_of(node))
        #print("Prev_layer:", prev_layer)

        sum = 0
        for prev in range(prev_nodes[0], prev_nodes[1]):
            #print("Node:", prev)
            sum += self._path_weights[(prev, node)] * self._node_weights[prev]
        sum += self._path_weights[("d", node)] * self._node_weights["d"]

        return sum

    def guess(self, inputs):
        self.forward_propagate(inputs[0])
        output_layer = self.get_layer_range(self._max_layer)
        res = []
        for node in range(output_layer[0], output_layer[1]):
            res.append(self._node_weights[node])

        return res

    def forward_propagate(self, inputs):
        """helper function - Forward propogates a single pair"""
        first_layer = self.get_layer_range(0)[1]
        #print("First loop")
        for node in range(first_layer):
            #print(node)
            self._node_weights[node] = inputs[node]

        #print("Second loop")
        for layer in range(1, self._max_layer + 1):
            #print("Layer: ", layer)
            nodes = self.get_layer_range(layer)
            for j in range(nodes[0], nodes[1]):
                #print("Node: ", j)
                inj = self.inj(j)
                self._node_weights[j] = logistic(inj)

    def back_propagate(self, pair):
        #print(pair)
        delta = {}
        output_layer = self.get_layer_range(self._max_layer)
        index = 0
        #print("First Loop")
        for node in range(output_layer[0], output_layer[1]):
            #print("output_layer:", output_layer)
            delta[node] = self.calculate_delta(node, delta, pair[1][index], True)
            index += 1
        #print("Second Loop")
        for layer in range(self._max_layer - 1, -1, -1):
            #print("Layer:", layer)
            nodes = self.get_layer_range(layer)
            for node in range(nodes[0], nodes[1]):
                #print("node:", node)
                delta[node] = self.calculate_delta(node, delta)

        for x,y in self._path_weights:
            self._path_weights[(x,y)] = self._path_weights[(x,y)] + (.1 * (self._node_weights[x] * delta[y]))

    def calculate_delta(self, node, delta, y = None, output = False):
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

    print("Correct: ", good)
    print("Incorrect: ", bad)
    print("Accuracy: ", good/(good + bad))

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
    nn = NeuralNetwork([30, 6, 1], 10000)
    nn.back_propagation_learning(real_training)

    for example in eval:
        guess = nn.guess(example)
        rounded = [round(item) for item in guess]
        print(example[1], ":", rounded)

    simple_accuracy(nn, eval)

if __name__ == "__main__":
    main()
