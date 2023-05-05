# Machine Learning with Tensorflow JS

## II Concepts
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


# Concepts

### Operations API
Mathematical operations in data. Low level. For creating machine learning models. Not easy, as it is the lowest level.
Provides optimizations thrugh GPU and webGL

### Layers API
To define complex models easily

TensorflowJS docs
`https://js.tensorflow.org/api/latest/`

### Tensors

In simple terms, tensors are a data structure. A generalization of vectors and matrices to potential higher dimensions, in JS terms *a tensor is a nested array*.

- Create a tensor: General function to create tensors. 
`tf.tensor([1,2,3,4], [2,2])` => In this case the values are expressed as an array and the shape [2,2] (2x2 grid) is specified.

Takes three parameters values(required), shape and dtype. The shape is derived from the values unless you specify it. dtype refers to the type of data.

- Create a scalar tensor (Single value)
`tf.scalar(1.234)`

- Create a 1 dimensional tensor
`tf.tensor1d([2,3,4])`

- Create a 2 dimensional tensor => Pass a nested array
`tf.tensor2d([[1,2,3,4], [2,3,4]])`

#### Definition
A tensor is a data structure which is a list of values with a data type (float, int, string)
A tensor has **dimensions**. For example 0d 1 - 1d [1,2,3] (an array / vector)- 2d [[1,2], [3,4,5]] (grid)  - 3d [[[1,2], [5,6]], [[3,4], [4,5]]] (cube)
A tenspor has a shape: rows, columns

##### Operations

- Tensors are immutable

- tf.add(a, b)
    Adds two tensors. If one is scalar, it will 'broadcast' the value. For example:
    ```
    const a = tf.scalar(5)
    const b = tf.tensor1d([1,2,3,4])

    a.add(b).print() // [6,7,8,9]

    // Different shape broadcast
    const a = tf.tensor1d([1,5]);
    const b = tf.tensor2d([[10, 20], [30, 40]]);

    tf.add(a, b) // [[11, 25], [31, 45]]
    ```

- tf.addStrict(a, b)
    Asserts both tensors have the same shape

##### Why using operations?
- For preparing data, like normalizing data (Range from 0 to 1)
- Build low-level models

#### Memory management
Tensors we use in TF are stored in WebGL.
In JS there's garbage colleciton. When using WebGL there's no garbage collection.

How to remove memory?

- tf.memory() -> Information about memory
- tf.dispose() -> Clean up a specific tensor
- tf.tidy() -> Wrapper function that will automatically tidy any tensors inside its scope after execution.
- tf.keep() -> Inform backend that it should keep a specific tensor in memory when inside a `tify` wrapper function.


## III Data preparation

### Linear regression
Plot a line on a dataset that has a certain correlation between two variables.
Steps
1. Prepare data
2. Create model => Line for he correlation
3. Train the model (Test the line against the data)
4. Test the model
5. Make predictions

#### 1. Data preparation
- Import data from csv
- Visualize data
- Normalize data

`kaggle.com` => public datasets to practice machine learning

To parse the CSV file, we have to run a local server for the html, becuse the browser doesn't have access to the file system. We will do that with the `http-server` npm package.

#### 2. Data visualization
tfjs-vis is a data visualizing data made for tensorflow js. Anythin run with the tfjs-vis library will appear on a right-hand visor on the browser.

#### 3. Data normalization
The `layers` api uses values from 0-1 , so we need to normalize our data before training, specially for more complex models.
Im machine leargning jargon, normalization is also called `Feature scaling`. We will use the min-max normalization, which is a formular that assigns
1 to the max value and 0 to the minimun of a dataset.

Formula:
**normlizedValue = (normalizedValue - dataMin) / (dataMax - dataMin)**

This formula divides the value minus the minimun value in the set byt the range in which the values exist (dataMax-dataMin)

To make sense of the predction data, we will need a denormalization process as well.