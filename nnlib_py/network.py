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

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_grad(x):
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return s * (1.0 - s)

    gradient_map = {
        relu.__func__: relu_grad.__func__,
        linear.__func__: linear_grad.__func__,
        sigmoid.__func__: sigmoid_grad.__func__
    }

class Losses:
    @staticmethod
    def mse(prediction, true_value):
        return np.mean((prediction - true_value) ** 2)
    
    @staticmethod
    def mse_grad(prediction, true_value):
        return 2 * (prediction - true_value) / prediction.size

    @staticmethod
    def cross_entropy(prediction, true_value):
        prediction = np.clip(prediction, 1e-15, 1.0 - 1.0e-15)
        return -np.sum(true_value * np.log(prediction)) / prediction.shape[0]
    
    @staticmethod
    def cross_entropy_grad(prediction, true_value):
        prediction = np.clip(prediction, 1e-15, 1.0 - 1.0e-15)
        return (-true_value / prediction) / prediction.shape[0]

    @staticmethod
    def softmax_cross_entropy(prediction_logits, true_value):
        m = prediction_logits.shape[0]
        shift_logits = prediction_logits - np.max(prediction_logits, axis=-1, keepdims=True)
        exps = np.exp(shift_logits)
        softmax_output = exps / np.sum(exps, axis=-1, keepdims=True)
        softmax_output = np.clip(softmax_output, 1e-15, 1.0 - 1.0e-15)
        loss = -np.sum(true_value * np.log(softmax_output)) / m
        return loss

    @staticmethod
    def softmax_cross_entropy_grad(prediction_logits, true_value):
        m = prediction_logits.shape[0]
        shift_logits = prediction_logits - np.max(prediction_logits, axis=-1, keepdims=True)
        exps = np.exp(shift_logits)
        softmax_output = exps / np.sum(exps, axis=-1, keepdims=True)
        return (softmax_output - true_value) / m

    gradient_map = {
        softmax_cross_entropy.__func__: softmax_cross_entropy_grad.__func__,
        mse.__func__: mse_grad.__func__,
        cross_entropy.__func__: cross_entropy_grad.__func__
    }

class LrDecays:
    @staticmethod
    def exponential_decay(current_lr, factor):
        return factor * current_lr

    @staticmethod
    def step_decay(current_lr, lr_drop, epoch, epoch_interval):
        if epoch % epoch_interval == 0:
            return current_lr - lr_drop
        else:
            return current_lr

    @staticmethod
    def linear_decay(current_lr, lr_drop, lr_min):
        return max(lr_min, current_lr - lr_drop)

    @staticmethod
    def cosine_decay(current_epoch, total_epochs, start_lr, min_lr):
        current_epoch = min(current_epoch, total_epochs)
        progress = current_epoch / total_epochs
        cosine_out = 0.5 * (1.0 + np.cos(np.pi * progress))
        lr = min_lr + (start_lr - min_lr) * cosine_out
        return lr

    @staticmethod
    def SGDR(current_epoch, start_cycle_epochs, cycle_multiplier, start_lr, min_lr):
        cycle_epochs = start_cycle_epochs
        epoch_in_cycle = current_epoch
        while epoch_in_cycle >= cycle_epochs:
            epoch_in_cycle -= cycle_epochs
            cycle_epochs *= cycle_multiplier
        progress = epoch_in_cycle / cycle_epochs
        cosine_out = 0.5 * (1.0 + np.cos(np.pi * progress))
        lr = min_lr + (start_lr - min_lr) * cosine_out
        return lr

    @staticmethod
    def inverse_time_decay(initial_lr, decay_rate, current_epoch, min_lr):
        return max(min_lr, initial_lr / (1.0 + decay_rate * current_epoch))

class WeightInitializers:
    @staticmethod
    def he(n_in, n_out): # goed voor relu
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

    @staticmethod
    def xavier(n_in, n_out): # goed voor tanh of sigmoid
        return np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)

    @staticmethod
    def random_small(n_in, n_out): # algemeen
        return np.random.randn(n_in, n_out) * 0.01

class PreTrainedLayer:
    def __init__(self, weights_location, biases_location, activation):
        self.weights = np.load(weights_location)
        self.biases = np.load(biases_location)
        self.activation = activation

    def forward(self, inputs):
        self.pre_activation = inputs @ self.weights + self.biases
        return self.activation(self.pre_activation)

class Layer:
    def __init__(self, n_in, n_out, activation, initializer):
        self.weights = initializer(n_in, n_out)
        self.biases = np.zeros((1, n_out))
        self.activation = activation

    def forward(self, inputs):
        self.cached_inputs = inputs
        self.pre_activation = inputs @ self.weights + self.biases
        return self.activation(self.pre_activation)

    def backward(self, incoming_gradient):
        activation_grad_func = Activations.gradient_map[self.activation]
        activation_gradient = activation_grad_func(self.pre_activation)
        gradient_after_activation = incoming_gradient * activation_gradient
        self.weight_gradient = self.cached_inputs.T @ gradient_after_activation
        self.bias_gradient = np.sum(gradient_after_activation, axis=0, keepdims=True)
        return gradient_after_activation @ self.weights.T


    def update(self, learning_rate):
        self.weights -= learning_rate * self.weight_gradient
        self.biases -= learning_rate * self.bias_gradient

class Network:
    def __init__(self, loss_function):
        self.layers = []
        self.num_layers = 0
        self.input_size = 0
        self.output_size = 0
        self.hidden_layer_size = 0
        self.cached_hidden_layer_size = 0
        self.screen = None
        self.font = None
        self.epoch = 0 # should be updated in the training loop, just for visualization
        self.loss = 0 # should be updated in the training loop, just for visualization
        self.current_lr = 0 # should be updated in the training loop, just for visualization
        self.loss_function = loss_function

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, prediction, true_value):
        loss_grad_func = Losses.gradient_map[self.loss_function]
        gradient = loss_grad_func(prediction, true_value)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def visualize(self, input_size, hidden_size, hidden_amount, output_size, window_width, window_height, method="stretched"):
        if not self.screen:
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))
            self.font = pygame.font.Font(None, 32)
        self.screen.fill((0,0,0))
        hor_side_offset = 40
        vert_side_offset = 60
        hor_gap = (window_width - 2 * hor_side_offset) / (hidden_amount + 1)
        biggest_node_amount = max(input_size, hidden_size, output_size)
        if method == "proportional":
            input_node_gap = (window_height - 2 * vert_side_offset) / (biggest_node_amount + 1)
            hidden_node_gap = input_node_gap
            output_node_gap = input_node_gap
        elif method == "stretched":
            input_node_gap = (window_height - 2 * vert_side_offset) / (input_size + 1)
            hidden_node_gap = (window_height - 2 * vert_side_offset) / (hidden_size + 1)
            output_node_gap = (window_height - 2 * vert_side_offset) / (output_size + 1)
        node_radius = 8
        for i in range(input_size):
            y = (window_height / 2) - (input_node_gap * ((input_size - 1) / 2)) + (i * input_node_gap)
            pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset, y), node_radius, 3)
        for q in range(hidden_amount):
            for i in range(hidden_size):
                y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (i * hidden_node_gap)
                pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset + hor_gap * (q + 1), y), node_radius, 3)
        for i in range(output_size):
            y = (window_height / 2) - (output_node_gap * ((output_size - 1) / 2)) + (i * output_node_gap)
            pygame.draw.circle(self.screen, (255,255,255), (hor_side_offset + (hidden_amount + 1) * hor_gap, y), node_radius, 3)
        for q in range(hidden_amount):
            if q == 0:
                start_x = hor_side_offset
                end_x = hor_side_offset + hor_gap
                for i in range(input_size):
                    start_y = (window_height / 2) - (input_node_gap * ((input_size - 1) / 2)) + (i * input_node_gap)
                    for p in range(hidden_size):
                        weight = self.layers[q].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (p * hidden_node_gap)
                        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
            else:
                start_x = hor_side_offset + q * hor_gap
                end_x = hor_side_offset + (q + 1) * hor_gap
                for i in range(hidden_size):
                    start_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (i * hidden_node_gap)
                    for p in range(hidden_size):
                        weight = self.layers[q].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (p * hidden_node_gap)
                        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
        for q in range(hidden_size):
            start_x = hor_side_offset + hor_gap * (hidden_amount)
            end_x = hor_side_offset + hor_gap * (hidden_amount + 1)
            start_y = (window_height / 2) - (hidden_node_gap * ((hidden_size - 1) / 2)) + (q * hidden_node_gap)
            for p in range(output_size):
                weight = self.layers[hidden_amount].weights[q, p]
                if weight > 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 134, 212)
                end_y = (window_height / 2) - (output_node_gap * ((output_size - 1) / 2)) + (p * output_node_gap)
                pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
        text = self.font.render(f"epoch: {self.epoch} | loss: {self.loss:.6f} | lr: {self.current_lr:.4f}", True, (255, 255, 255))
        self.screen.blit(text, (window_width / 2 - text.get_width() / 2, (vert_side_offset - text.get_height()) / 2))
        pygame.display.update()

    def check_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return