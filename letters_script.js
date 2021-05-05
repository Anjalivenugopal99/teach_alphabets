import {LMnistData} from './letters_data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;
const NUM_DATASET_ELEMENTS = 124800//10000//10000;//change
 
const TRAIN_TEST_RATIO = 5 / 6;
 
const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
const NUM_CHANNELS = 1


function getModel() {
    
    // In the space below create a convolutional neural network that can classify the 
    // images of articles of clothing in the Fashion MNIST dataset. Your convolutional
    // neural network should only use the following layers: conv2d, maxPooling2d,
    // flatten, and dense. Since the Fashion MNIST has 10 classes, your output layer
    // should have 10 units and a softmax activation function. You are free to use as
    // many layers, filters, and neurons as you like.  
    // HINT: Take a look at the MNIST example.
    model = tf.sequential();
    
    // YOUR CODE HERE
    model.add(tf.layers.conv2d({inputShape: [28, 28, NUM_CHANNELS], kernelSize: 5, filters: 8,padding: 'same', activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    //model.add(tf.layers.batchNormalization())
    //model.add(tf.layers.dropout({ rate: 0.25 }))
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 5, padding: 'same', activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	model.add(tf.layers.dense({units: 26, activation: 'softmax'}));

    
    
    // Compile the model using the categoricalCrossentropy loss,
    // the tf.train.adam() optimizer, and accuracy for your metrics.
    model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']});
    
    return model;
}

async function train(model, data) {
        
    // Set the following metrics for the callback: 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
    const metrics =  ['loss', 'val_loss', 'acc', 'val_acc'];

        
    // Create the container for the callback. Set the name to 'Model Training' and 
    // use a height of 1000px for the styles. 
    const container = { name: 'Model Training', styles: { height: '640px' } };  
    
    
    // Use tfvis.show.fitCallbacks() to setup the callbacks. 
    // Use the container and metrics defined above as the parameters.
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 7000; //change to 100 //83200
    const TEST_DATA_SIZE = 3000; //change to 50//41600
    
    // Get the training batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(NUM_TRAIN_ELEMENTS);
		return [
			d.xs.reshape([NUM_TRAIN_ELEMENTS, 28, 28, NUM_CHANNELS]),
			d.labels
		];
	});
    

    
    // Get the testing batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(NUM_TEST_ELEMENTS);
		return [
			d.xs.reshape([NUM_TEST_ELEMENTS, 28, 28, NUM_CHANNELS]),
			d.labels
		];
	});
    //console.log("train"+"test"+trainXs[0]+trainYs[0]);
    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
        // callbacks:{
        //     onEpochEnd: async(epoch, logs) =>{
        //         console.log("Epoch: " + epoch + " Loss: " + logs.loss);
        //     }}
        });
    
}

function setPosition(e){
    pos.x = e.clientX-100;
    pos.y = e.clientY-100;
}
    
function draw(e) {
    if(e.buttons!=1) return;
    ctx.beginPath();
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    rawImage.src = canvas.toDataURL('image/png');
}
    
function erase() {
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,280,280);
}
    
function save() {
    var raw = tf.browser.fromPixels(rawImage,1);
    var resized = tf.image.resizeBilinear(raw, [28,28]);
    var tensor = resized.expandDims(0);
    
    var prediction = model.predict(tensor);
    console.log("prediction"+ prediction);
    var pIndex = tf.argMax(prediction, 1).dataSync();
    console.log("pIndex"+ pIndex);
    
    var classNames = ["A", "B", "C", 
                      "D", "E", "F", "G",
                      "H", "I", "J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];
            
            
    alert(classNames[pIndex]);
}
    
function init() {
    canvas = document.getElementById('canvas');
    rawImage = document.getElementById('canvasimg');
    ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,280,280);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mousedown", setPosition);
    canvas.addEventListener("mouseenter", setPosition);
    saveButton = document.getElementById('sb');
    saveButton.addEventListener("click", save);
    clearButton = document.getElementById('cb');
    clearButton.addEventListener("click", erase);
}


async function run() {
    const data = new LMnistData();
    await data.load();
    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
    await train(model, data);
    //await model.save('downloads://my_model');
    init();
    alert("Training is done, try classifying your drawings!");
}

document.addEventListener('DOMContentLoaded', run);
