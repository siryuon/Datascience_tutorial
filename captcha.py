import random
import math
import matplotlib
import matplotlib.pyplot as plt

from collections import Counter


def dot(v, w):
    return sum(v_i * w_i
                for v_i, w_i in zip(v, w))

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    """((����) list�� (������) list�� (weight)�� list)���� ������ �Ű���� �Է� �ް� ���������� �����Ͽ� ����� ��ȯ"""
    outputs=[]

    #�� ������ ���
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                for neuron in layer]
        outputs.append(output)
        #�׸��� �̹� ���� ����� ���� ���� �Էº����� �ȴ�.
        input_vector = output
    
    return outputs

def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    output_deltas = [output * (1 - output) * (output - target)
                    for output, target in zip(outputs, targets)]
    
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            #�� ������ ��ȭ���� j ��° �Է� ���� ����Ͽ� j���� weight�� ����
            output_neuron[j] -= output_deltas[i] * hidden_output
    
    #���������� �������� �ڷ� ����
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                    dot(output_deltas, [n[i] for n in output_layer])
                    for i, hidden_output in enumerate(hidden_outputs)]

    #�������� �������� weight�� ����
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

raw_digits = [
          """11111
             1...1
             1...1
             1...1
             11111""",

          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",

          """11111
             ....1
             11111
             1....
             11111""",

          """11111
             ....1
             11111
             ....1
             11111""",

          """1...1
             1...1
             11111
             ....1
             ....1""",

          """11111
             1....
             11111
             ....1
             11111""",

          """11111
             1....
             11111
             1...1
             11111""",

          """11111
             ....1
             ....1
             ....1
             ....1""",

          """11111
             1...1
             11111
             1...1
             11111""",

          """11111
             1...1
             11111
             ....1
             11111"""]


def make_digit(raw_digit):
    return [1 if c == '1' else 0
            for row in raw_digit.split("\n")
            for c in row.strip()]


inputs = map(make_digit, raw_digits)
targets = [[1 if i == j else 0 for i in range(10)]
           for j in range(10)]

random.seed(0)
input_size = 25
num_hidden = 5
output_size = 10

hidden_layer = [[random.random() for __ in range(input_size + 1)]
                for __ in range(num_hidden)]
output_layer = [[random.random() for __ in range(num_hidden + 1)]
                for __ in range(output_size)]

network = [hidden_layer, output_layer]

for __ in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)

def predict(input):
    return feed_forward(network, input)[-1]

