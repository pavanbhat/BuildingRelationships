"""
Intuit Challenge - Building Relationships
Author: Pavan Bhat (pxb8715.rit.edu)
"""
import csv
import math


class network:
    '''
    Class network to create the nodes of the neural network
    '''
    __slots__ = 'numHiddenNodes', 'inputNodes', 'outputNodes', 'hiddenNodes', 'biasNodes', 'v', 'layer', 'weights'

    def __init__(self, numHiddenNodes, filename):
        '''
        Constructor to initialize the neural net
        :param numHiddenNodes:
        '''
        self.inputNodes = []
        self.outputNodes = [1, 2, 3, 4]
        self.hiddenNodes = []
        self.biasNodes = []
        self.v = [0 for i in range(numHiddenNodes + 9)]
        self.layer = [[] for i in range(3)]
        self.weights = []
        self.setup(numHiddenNodes)
        self.initWeights(filename)

    def setup(self, numHiddenNodes):
        '''
        Set up the neural net architecture
        :param numHiddenNodes: number of hidden layer nodes
        :return:
        '''
        # set up node numbers in the layers

        self.biasNodes.append(5)

        for i in range(numHiddenNodes):
            self.hiddenNodes.append(i + 6)
        self.biasNodes.append(6 + numHiddenNodes)
        self.inputNodes.append(7 + numHiddenNodes)
        self.inputNodes.append(8 + numHiddenNodes)

        # create layers and add nodes to them
        self.layer[0].extend(self.inputNodes)
        self.layer[0].append(self.biasNodes[1])
        self.layer[1].extend(self.hiddenNodes)
        self.layer[1].append(self.biasNodes[0])
        self.layer[2].extend(self.outputNodes)

        # assign values to bias nodes
        self.v[self.biasNodes[0]] = 1
        self.v[self.biasNodes[1]] = 1

    def initWeights(self, filename):
        '''
        Initialize the weights
        :param filename: file to read the weights from
        :return:
        '''
        file = open(filename)
        dataset = csv.reader(file)

        for line in dataset:
            a = []
            for j in line:
                a.append(float(j))
            self.weights.append(a)

def test(weights, obj):
    '''
    Test function to iterate over the test samples and classify
    :param weights: weights in the neural net
    :param obj:
    :return:
    '''
    name = 'purchasing_power.csv'
    file = open(name)
    data_set = csv.reader(file)

    # input attributes
    x = []

    # output vector
    y = []

    # confusion matrix
    matrix = [[0 for i in range(4)] for j in range(4)]

    # read the rows in dataset and store the values of attributes in arrays
    for line in data_set:
        if line != []:
            x.append([float(line[0]), float(line[1])])
            y.append(int(line[2]))

    classified = []

    # test the inputs
    for i in range(len(x)):

        maxval = 0
        # Propagate the input forward to compute the outputs
        for inp, index in zip(obj.inputNodes, range(2)):
            obj.v[inp] = x[i][index]

        for l in range(1, len(obj.layer)):
            for j in obj.layer[l]:
                if j not in obj.biasNodes:
                    sum_in = 0.0
                    for k in obj.layer[l - 1]:
                        sum_in += (weights[k][j] * obj.v[k])
                    obj.v[j] = hypothesis(sum_in)

        # assign the class to the test samples
        for k in obj.outputNodes:
            if maxval < obj.v[k]:
                maxval = obj.v[k]
                classifier = k
        classified.append(classifier)

    # Calculate the confusion matrix and compute the overall profit
    sum = 0
    profit = 0
    cost = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]
    for i, j in zip(classified, range(0, len(y))):
        matrix[i-1][y[j]-1] += 1
        profit += cost[i-1][y[j]-1]
        if i-1 == y[j]-1:
            sum += 1

    print('Recognition rate (% correct) = ', ((sum/len(y))*100))
    print('Likelihood of relationship obtained = ', round(profit*0.01, 2))

    print('\nConfusion Matrix: ')
    print('\tActual -->\t\tClass 1\tClass 2\t\tClass 3\tClass 4')
    for i in range(4):
        if i == 0:
            print('Assigned as Class 1  : ', end="")
        elif i == 1:
            print('Assigned as Class 2  : ', end="")
        elif i == 2:
            print('Assigned as Class 3  : ', end="")
        elif i == 3:
            print('Assigned as Class 4  : ', end="")
        for j in range(4):
            print(matrix[i][j], ' \t\t', end="")
        print()


def hypothesis(sum_in):
    '''
    Sigmoid activation function to compute the hypothesis
    :param attr1: the first attribute
    :param attr2:  the second attribute
    :param weights: weights associated with the attributes and bias
    :return: the current value of hypothesis calculated
    '''
    return 1 / (1 + math.exp(-sum_in))


def main():
    '''
    The Main method
    :return:
    '''
    numHiddenNodes = int(input("Enter number of hidden layer nodes: (ideally = 5) "))

    for i in [1000, 5000]:
        name = 'weights'+str(i)+'.csv'
        print('\n\n ---- Epochs = ', i, ' ----')
        n = network(numHiddenNodes, name)
        test(n.weights, n)


main()
