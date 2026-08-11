import matplotlib.pyplot as plt
import numpy as np

from classiq import *
from classiq.applications.qsvm.qsvm import QSVM, QuantumKernelEvaluator
from classiq.applications.qsvm.quantum_feature_maps import pauli_feature_map
from classiq.open_library.functions.variational import inplace_encode_on_bloch


import random

seed = 0
random.seed(seed)
np.random.seed(seed)

# Importing functions used for this demo, to generate random linearly separable data
from classiq.applications.qsvm.qsvm_data_generation import (
    data_dict_to_data_and_labels,
    generate_data,
)

# Generate sample data:
sources = np.array([[1.23016026, 1.72327701], [3.20331931, 5.32365722]])

training_input: dict = generate_data(sources=sources)
test_input: dict = generate_data(sources=sources)
predict_input, predict_real_labels = data_dict_to_data_and_labels(
    generate_data(sources=sources)
)

# Prepare and define `train_input` and `test_input` datasets consisting of data and labels
TRAIN_DATA_1, TRAIN_LABELS_1 = data_dict_to_data_and_labels(training_input)
TEST_DATA_1, TEST_LABELS_1 = data_dict_to_data_and_labels(test_input)

# Prepare and define `predict_input`
PREDICT_DATA_1 = predict_input


def normalize(data: np.ndarray, range: tuple) -> np.ndarray:
    return (2 * data - range[1] - range[0]) / (range[1] - range[0])


RANGE = (0, 6 * np.pi)
TRAIN_DATA_1 = normalize(TRAIN_DATA_1, RANGE)
TEST_DATA_1 = normalize(TEST_DATA_1, RANGE)
PREDICT_DATA_1 = normalize(PREDICT_DATA_1, RANGE)

plot_range = (-1, -0.4)
colors = {0: "blue", 1: "orange"}
for i in range(TRAIN_DATA_1.shape[0]):
    plt.scatter(*TRAIN_DATA_1[i, :].T, c=colors[TRAIN_LABELS_1[i]])
plt.title("Training Data")
plt.xlim(plot_range)
plt.ylim((-1, 0))
plt.show()

for i in range(TRAIN_DATA_1.shape[0]):
    plt.scatter(*TEST_DATA_1[i, :].T, c=colors[TEST_LABELS_1[i]])
plt.title("Test Data")
plt.xlim(plot_range)
plt.ylim((-1, 0))
plt.show()

plt.scatter(*PREDICT_DATA_1.T)
plt.title("Prediction Data (unlabeled)")
plt.xlim((-1, -0.4))
plt.ylim((-1, -0.2))
plt.show()

# Build a quantum support vector machine model
bloch_num_qubits = int(np.ceil(np.log2(TRAIN_DATA_1.shape[1])))
qsvm_model = QSVM(feature_map=inplace_encode_on_bloch, num_qubits=bloch_num_qubits)

qprog_bloch = qsvm_model.get_qprog(data_dim=TRAIN_DATA_1.shape[1])
show(qprog_bloch)

# Train the model
qsvm_model.train(TRAIN_DATA_1, TRAIN_LABELS_1)

# Check the test score
test_score, test_predicted_labels = qsvm_model.test(TEST_DATA_1, TEST_LABELS_1)

# Predict labels
predicted_labels = qsvm_model.predict(PREDICT_DATA_1)

# Printing tests result
print(f"Testing success ratio: {test_score}")
print()
# Printing predictions
print("Prediction from datapoints set:")
print(f"  ground truth: {predict_real_labels}")
print(f"  prediction:   {predicted_labels}")
print(
    f"  success rate: {100 * np.count_nonzero(predicted_labels == predict_real_labels) / len(predicted_labels)}%"
)

plt.figure()
for i in range(PREDICT_DATA_1.shape[0]):
    plt.scatter(*PREDICT_DATA_1[i, :].T, c=colors[predicted_labels[i]])
plt.title("Prediction Data")
plt.xlim(plot_range)
plt.ylim((-1, 1 + 0.3))
plt.show()