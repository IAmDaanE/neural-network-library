import numpy as np
import pygame

class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_grad(x):
        return np.ones_like(x)

    gradient_map = {
        relu.__func__: relu_grad.__func__,
        linear.__func__: linear_grad.__func__,
    }

class Losses:
    @staticmethod
    def mse(prediction, true_value):
        return np.mean((prediction - true_value) ** 2)
    
    @staticmethod
    def mse_grad(prediction, true_value):
        return 2 * (prediction - true_value) / prediction.size

    gradient_map = {
        mse.__func__: mse_grad.__func__
    }

class Layer:
    def __init__(self, n_in, n_out, activation):
        self.weights = np.random.randn(n_in, n_out) * 0.1
        self.biases = np.zeros((1, n_out))
        self.activation = activation

    def forward(self, inputs):
        self.cached_inputs = inputs
        self.pre_activation = inputs @ self.weights + self.biases
        return self.activation(self.pre_activation)

    def backward(self, incoming_gradient):
        activation_gradient = Activations().gradient_map[self.activation](self.pre_activation)
        gradient_after_activation = incoming_gradient * activation_gradient
        self.weight_gradient = self.cached_inputs.T @ gradient_after_activation
        self.bias_gradient = np.sum(gradient_after_activation, axis=0, keepdims=True)
        return gradient_after_activation @ self.weights.T

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weight_gradient
        self.biases -= learning_rate * self.bias_gradient

class Network:
    def __init__(self, loss_function=Losses().mse, input_size=0, hidden_size=0, hidden_amount=0, output_size=0, window_width=0, window_height=0):
        self.layers = []
        self.loss_function = loss_function
        self.num_layers = 0
        self.input_size = 0
        self.output_size = 0
        self.hidden_layer_size = 0
        self.cached_hidden_layer_size = 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_amount = hidden_amount
        self.output_size = output_size
        self.window_width = window_width
        self.window_height = window_height
        self.screen = None
        self.window_width = window_width
        self.window_height = window_height

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, prediction, true_value):
        gradient = Losses().gradient_map[self.loss_function](prediction, true_value)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def visualize(self):
        if not self.screen:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return 
        self.screen.fill((0,0,0))
        hor_side_offset = 40
        vert_side_offset = 15
        hor_gap = (self.window_width - 2 * hor_side_offset) / (self.hidden_amount + 1)
        biggest_node_amount = max(self.input_size, self.hidden_size, self.output_size)
        node_gap = (self.window_height - 2 * vert_side_offset) / (biggest_node_amount + 1)
        node_radius = 8
        for i in range(self.input_size):
            y = self.window_height / 2 - node_gap * (self.input_size / 2) + i * node_gap
            pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset, y), node_radius, 3)
        for q in range(self.hidden_amount):
            for i in range(self.hidden_size):
                y = self.window_height / 2 - node_gap * (self.hidden_size / 2) + i * node_gap
                pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset + hor_gap * (q + 1), y), node_radius, 3)
        for i in range(self.output_size):
            y = self.window_height / 2 - node_gap * (self.output_size / 2) + i * node_gap
            pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset + (self.hidden_amount + 1) * hor_gap, y), node_radius, 3)
        for q in range(self.hidden_amount - 1):
            if q == 0:
                start_x = hor_side_offset
                end_x = hor_side_offset + hor_gap
                for i in range(self.input_size):
                    start_y = self.window_height / 2 - node_gap * (self.input_size / 2) + i * node_gap
                    for p in range(self.hidden_size):
                        weight = self.layers[q].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = self.window_height / 2 - node_gap * (self.hidden_size / 2) + p * node_gap
                        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
            else:
                start_x = hor_side_offset + q * hor_gap
                end_x = hor_side_offset + (q + 1) * hor_gap
                for i in range(self.hidden_size):
                    start_y = self.window_height / 2 - node_gap * (self.hidden_size / 2) + i * node_gap
                    for p in range(self.hidden_size):
                        weight = self.layers[q].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = self.window_height / 2 - node_gap * (self.hidden_size / 2) + p * node_gap
                        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
        for q in range(self.hidden_size):
            start_x = hor_side_offset + hor_gap * (self.hidden_amount)
            end_x = hor_side_offset + hor_gap * (self.hidden_amount + 1)
            start_y = self.window_height / 2 - node_gap * (self.hidden_size / 2) + q * node_gap
            for p in range(self.output_size):
                weight = self.layers[self.hidden_amount + 1].weights[q, p]
                if weight > 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 134, 212)
                end_y = self.window_height / 2 - node_gap * (self.output_size / 2) + p * node_gap
                pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
        pygame.display.update()