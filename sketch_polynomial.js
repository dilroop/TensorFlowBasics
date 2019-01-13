// Polynomial Line 

let mouse_x = [];
let mouse_y = [];

// let m,b; // line the varibles
let a, b, c, d; // line the varibles

let start_x = -1;
let end_x = 1;
let start_y = 1;
let end_y = -1;

const learingRate = 0.5;
const optimizer = tf.train.adam(learingRate)

function setup() {
    createCanvas(400, 400);
    a = tf.variable(tf.scalar(random(start_x, end_x)));
    b = tf.variable(tf.scalar(random(start_x, end_x)));
    c = tf.variable(tf.scalar(random(start_x, end_x)));
    d = tf.variable(tf.scalar(random(start_x, end_x)));
}

function mousePressed() {
    let x = map(mouseX, 0, width, start_x, end_x);
    let y = map(mouseY, 0, height, start_y, end_y);
    mouse_x.push(x);
    mouse_y.push(y);
}

function draw() {
    tf.tidy(function () {

        if (mouse_x.length > 0) {
            const ysAsTensor = tf.tensor1d(mouse_y)
            optimizer.minimize(() => loss(predict(mouse_x), ysAsTensor));
        }

        background(0);
        stroke(255);
        strokeWeight(7);

        for (let i = 0; i < mouse_x.length; i++) {
            let px = map(mouse_x[i], start_x, end_x, 0, width);
            let py = map(mouse_y[i], start_x, end_x, height, 0);
            point(px, py);
        }

        const curveX = [];
        for (let i = start_x; i < end_x; i += 0.005) {
            curveX.push(i)
        }

        const ysTidy = tf.tidy(() => predict(curveX))
        let curveY = ysTidy.dataSync();

        beginShape();
        noFill();
        stroke(255);
        strokeWeight(1)

        for (let i = 0; i < curveX.length; i++) {
            let x = map(curveX[i], start_x, end_x, 0, width);
            let y = map(curveY[i], start_x, end_x, height, 0);
            vertex(x, y);
        }
        endShape();

    })

    console.log(tf.memory().numTensors)
}

function predict(xsFromMouse) {
    const xsAsTensor = tf.tensor1d(xsFromMouse);
    // y = ax^2 + bx + c
    const ysPredictions = xsAsTensor.pow(tf.scalar(3)).mul(a)
        .add(b.square().mul(xsAsTensor))
        .add(c.mul(xsAsTensor))
        .add(d)
    return ysPredictions;
}

function loss(preditions, labels) {
    return preditions.sub(labels).square().mean();
}

// function predict(xs) {
//     const tfxs = tf.tensor1d(xs);
//     // y = mx + b
//     const ys = tfxs.mul(m).add(b)
//     return ys;
// }