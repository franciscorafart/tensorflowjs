# Machine Learning with Tensorflow JS

### Concepts
- Machine learning is a subcategory of artificial intelligence.
- Perform tasks without specific programmed rules => Behavior is based in data instead of rules.
- Most ML tasks are predictions

#### Supervised learning
In supervised learning or classification ML we have a clean data set of the information we want to classify. In unsupervised learning we don't a dataset for training, we try to identify patterns in the available data.

#### Training
To train an ML model, we split the dataset in two: One set for training, the other for testing that the model is working well with data that wasn't used in training.

**Overfitting**: Training a model that works really well with a dataset, but doesn't work to predict well new data.

To determine how good a model works we use a **Loss Function**. It calculates the distance from each point to the prediction line in the model graph, the calculate an avarage to obtain a loss value.

Typical loss function: **MSE (Mean square Error)**

#### Artificial neural networks overview
- Artificail neural networks are used for supervised learning ML
- Inspired by the biological neural networks in the brain
- Arificial neurons are implemented in software and do not correlate to the biological one.

- A neural network has inputs and outputs.
- The process of the neural network can be trained with **backpropagation**
- For a programmer the neural network is a black box, it's internal dynamics have little meaning.
- Network configration is important => We can't use the same neural network to solve the same problems.

#### Artificial Neuron (Node)
Inputs (n) => Activation Function => Other Neurons / Nodes =>

It has any number of inputs and one output. The inputs are weighted and then summed (with an optional bias) => Activation Function

The activation function determines wether the neuron produces an output or what output it produces (binary or sigmoid function)

#### Neural network
Neural networks are organized by layers: Input, output, and inner layers.
When training the neural network compares the output node result with the expected result and then, applies backpropagation to correct the weights of every node and bias.

TEnsorflow playground: `https://playground.tensorflow.org/`