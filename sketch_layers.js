const model = tf.sequential();

const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
});
model.add(hidden);

const output = tf.layers.dense({
    units: 3,
    inputShape: [4],
    activation: 'sigmoid'
});
model.add(output);

const learningRate = 0.1;
const sgdOptimizer = tf.train.sgd(learningRate)

model.compile({
    optimizer: sgdOptimizer,
    loss: 'meanSquaredError'
});
// Training
const trainingInputData = tf.tensor2d([
    [0.92,0.36],
    [0.21,0.45]
]); 

const trainingOutputData = tf.tensor2d([
    [0.92,0.36, 0.19],
    [0.72,0.46, 0.29]
]); 

model.fit(trainingInputData, trainingOutputData, {epoch: 5})
        .then((response) => console.log(response.history.loss[0]));

// Predict
// const inputs = tf.tensor2d([[0.92,0.36]]); 
// const outputs = model.predict(inputs);
//outputs.print()