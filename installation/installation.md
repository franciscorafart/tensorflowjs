## Installation
https://www.tensorflow.org/js/tutorials/setup

### ML in the browser
Using tensorflow in the browser allows us to access user interaction, gives access to input devices such as camera and microhpone. Also, the user's device runs the code, to we don't need to run an expensive centralized
    1. Installing via script tag. Import it like in the `tensorflow.html` file and then check for the `tf` object in the console. It should be available globally.

## WebGL optimizations
WebGL is a 3D graphics API for the browser. Tensorflow uses the graphics card through WebGL for parallel matrix opreations, instead of using the CPU that does a single operation at a time.
Tensorflow provides a fallback when webgl is not available (few browser\)

`tf.getBackend()` to test which backend tensorflow is using in the browser. In the browser it's either `webgl` or `cpu`. In Node it can be `node`.

### ML in NodeJS
- It allows to integrate into existing backend applications written in Node instead of doing it with another language like Python.
- Make command line scripts

To install
`npm install @tensorflow/tfjs-node`
This is a JS wrapper of the original Python tensorflow.

### Tensorflow in React Native
Currently Tensrflow is available in react native using the react native webgl `rn-webgl`
https://tech.courses/tensorflow-js-react-native/
Template starter project for React Native: `https://github.com/Polarisation/react-native-template-tfjs`