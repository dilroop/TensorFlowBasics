// Tensor basics
function setup() {
    noCanvas()

    var x = 5;
    var y = 2;

    const values = []
    for (let i = 0; i < x * y; i++) {
        values[i] = random(0, 100)
    }

    const shape = [x, y];
    const shape2 = [2, 5]

    const a = tf.tensor2d(values, shape, 'int32');
    const b = tf.tensor2d(values, shape2, 'int32');

    b.print();
    
    const bb = b.transpose();

    bb.print()
}