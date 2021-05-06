import {LMnistData} from './letters_data.js';
var model;
const NUM_DATASET_ELEMENTS = 124800//10000//10000;//change
 
const TRAIN_TEST_RATIO = 5 / 6;
 
const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
const NUM_CHANNELS = 1


function getModel() {
    
    model = tf.sequential();
    model.add(tf.layers.conv2d({inputShape: [28, 28, NUM_CHANNELS], kernelSize: 5, filters: 8,padding: 'same', activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 5, padding: 'same', activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	model.add(tf.layers.dense({units: 26, activation: 'softmax'}));

   
    model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']});
    
    return model;
}

async function train(model, data) {
        
    const metrics =  ['loss', 'val_loss', 'acc', 'val_acc'];

        
    // Create the container for the callback. Set the name to 'Model Training'.
    const container = { name: 'Model Training', styles: { height: '640px' } };  
    
    
    // tfvis.show.fitCallbacks() to setup the callbacks. 
    // container and metrics defined above as the parameters.
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 7000; //change to 100 //83200
    const TEST_DATA_SIZE = 3000; //change to 50//41600
    
    // Get the training batches and resize them. 
    //  tf.tidy() - to clean up all the intermediate tensors.
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(NUM_TRAIN_ELEMENTS);
		return [
			d.xs.reshape([NUM_TRAIN_ELEMENTS, 28, 28, NUM_CHANNELS]),
			d.labels
		];
	});
    
    const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(NUM_TEST_ELEMENTS);
		return [
			d.xs.reshape([NUM_TEST_ELEMENTS, 28, 28, NUM_CHANNELS]),
			d.labels
		];
	});
   
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