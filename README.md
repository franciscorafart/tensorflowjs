# Machine Learning with Tensorflow JS

## Concepts
- Machine learning is a subcategory of artificial intelligence.
- It performs tasks without specific programmed rules, meaning behavior is based on data instead of rules.
- Most machine learning tasks involve predictions.

### Supervised learning
In supervised learning or classification machine learning, we have a clean dataset of the information we want to classify. In unsupervised learning, we don't have a dataset for training, and we try to identify patterns in the available data.

### Training
To train a machine learning model, we split the dataset in two: one set for training and another for testing to verify that the model is working well with data that wasn't used in training.

**Overfitting**: Training a model that works really well with a dataset but doesn't predict new data well.

To determine how well a model works, we use a **loss function**. It calculates the distance from each point to the prediction line in the model graph and then calculates an average to obtain a loss value.

Typical loss function: **MSE (Mean square Error)**

### Artificial neural networks overview
- Artificial neural networks are used for supervised learning machine learning.
- They are inspired by the biological neural networks in the brain.
- Artificial neurons are implemented in software and do not correlate to biological ones.
- A neural network has inputs and outputs.
- The process of the neural network can be trained with **backpropagation**.
- For a programmer, the neural network is a black box, and its internal dynamics have little meaning.
- Network configuration is important; we can't use the same neural network to solve different problems.

### Artificial Neuron (Node)
Inputs (n) => Activation Function => Other Neurons / Nodes =>

It has any number of inputs and one output. The inputs are weighted and then summed (with an optional bias) => Activation Function

The activation function determines whether the neuron produces an output or what output it produces (binary or sigmoid function).

### Neural network
Neural networks are organized by layers: Input, output, and inner layers. When training the neural network, it compares the output node result with the expected result and then applies backpropagation to correct the weights of every node and bias.

Tensorflow playground: `https://playground.tensorflow.org/`
