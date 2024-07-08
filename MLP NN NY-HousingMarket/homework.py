import numpy as np
import pandas as pd

def relu(X):
    return np.maximum(0, X)

def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

def categorical_cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def preprocess_features(df):
    df = df[['BATH', 'PROPERTYSQFT', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LATITUDE', 'LONGITUDE']].copy()
    df['ADMINISTRATIVE_AREA_LEVEL_2'] = pd.factorize(df['ADMINISTRATIVE_AREA_LEVEL_2'])[0]
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def preprocess_labels(df):
    return pd.get_dummies(df['BEDS']).values

def randomize_data(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, decay_rate, l2_reg):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.l2_reg = l2_reg

        self.weights = []
        self.biases = []

        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2 / sizes[i]))
            self.biases.append(np.zeros(sizes[i+1]))

    def adjust_learning_rate(self, epoch):
        return self.learning_rate * (self.decay_rate ** epoch)

    def forward_propagation(self, X):
        activations = []
        for i in range(len(self.weights)):
            X = relu(np.dot(X, self.weights[i]) + self.biases[i])
            activations.append(X)
        return softmax(X)

    def backward_propagation(self, X, y, lr):
        num_samples = X.shape[0]
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]

        activations = [X]
        for i in range(len(self.weights)):
            X = relu(np.dot(X, self.weights[i]) + self.biases[i])
            activations.append(X)

        error = activations[-1] - y
        for i in range(len(self.weights)-1, -1, -1):
            grad_weights[i] = np.dot(activations[i].T, error) / num_samples + self.l2_reg * self.weights[i]
            grad_biases[i] = np.sum(error, axis=0) / num_samples
            error = np.dot(error, self.weights[i].T) * (activations[i] > 0)

        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad_weights[i]
            self.biases[i] -= lr * grad_biases[i]

    def train(self, X_train, y_train, epochs, batch_size):
        for epoch in range(epochs):
            lr = self.adjust_learning_rate(epoch)
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                self.backward_propagation(X_train[start:end], y_train[start:end], lr)
            loss = categorical_cross_entropy_loss(y_train, self.forward_propagation(X_train))

    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)


train_data = pd.read_csv('train_data.csv')
train_labels = pd.read_csv('train_label.csv')
test_data = pd.read_csv('test_data.csv')

X_train = preprocess_features(train_data).values
y_train = preprocess_labels(train_labels)
X_test = preprocess_features(test_data).values

np.random.seed(42)
X_train_random, y_train_random = randomize_data(X_train, y_train)

input_size = X_train.shape[1]
hidden_sizes = [256, 128, 64]
output_size = y_train.shape[1]
epochs = 150
batch_size = 128
learning_rate = 0.04
decay_rate = 0.99
l2_reg = 0.001

model = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate, decay_rate, l2_reg)
model.train(X_train_random, y_train_random, epochs, batch_size)

y_pred = model.predict(X_test)
y_pred = y_pred + 1
output = pd.DataFrame({'BEDS': y_pred})
output.to_csv('output.csv', index=False)