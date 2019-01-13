const model = tf.sequential();
const actHiddenFn = 'relu'
const actOutputFn = 'linear'
const hidden1 = tf.layers.dense({
    units: 2,
    inputShape: [2],
    activation: actHiddenFn
});
model.add(hidden1);

const hidden2 = tf.layers.dense({
    units: 2,
    inputShape: [2],
    activation: actHiddenFn
});
model.add(hidden2);

const output = tf.layers.dense({
    units: 1,
    inputShape: [2],
    activation: actOutputFn
});
model.add(output);

const learningRate = 0.5;
const sgdOptimizer = tf.train.sgd(learningRate)

model.compile({
    optimizer: sgdOptimizer,
    loss: 'meanSquaredError'
});
// Training
const trainingInputData = tf.tensor2d([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]);

const trainingOutputData = tf.tensor2d([
    [0],
    [1],
    [1],
    [0]
]);

train().then(() => {
    console.log("Training Complete")
    // Predict
    const inputs = tf.tensor2d([[1,0]]); 
    const outputs = model.predict(inputs);
    outputs.print()
})

async function train() {
    for (let i = 0; i < 500; i++) {
        const response = await model.fit(trainingInputData, trainingOutputData, { epoch: 5, shuffle: true })
        console.log(response.history.loss[0])
    }
}

// Predict
// const inputs = tf.tensor2d([[0.92,0.36]]); 
// const outputs = model.predict(inputs);
//outputs.print()