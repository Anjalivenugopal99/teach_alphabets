

 const IMAGE_SIZE = 28*28;
 const NUM_CLASSES = 26;
 const NUM_DATASET_ELEMENTS = 124800//10000;//change
 
 const TRAIN_TEST_RATIO = 5 / 6;
 
 const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
 const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
 
 const MNIST_IMAGES_SPRITE_PATH ='data_img.png';
 const MNIST_LABELS_PATH ='labels_img';

 const NUM_CHANNELS = 1

 
 /**
  * A class that fetches the sprited MNIST letters dataset and returns shuffled batches.
  *
  *
  */
 export class LMnistData {
   constructor() {
     this.shuffledTrainIndex = 0;
     this.shuffledTestIndex = 0;
   }
 
   async load() {
     // Make a request for the MNIST sprited image.
     const img = new Image();
     const canvas = document.createElement('canvas');
     const ctx = canvas.getContext('2d');
     const imgRequest = new Promise((resolve, reject) => {
       img.crossOrigin = '';
       img.onload = () => {
         img.width = img.naturalWidth;
         img.height = img.naturalHeight;
 
         const datasetBytesBuffer =
             new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4*NUM_CHANNELS);
 
         const chunkSize    = Math.floor(NUM_TEST_ELEMENTS * 0.15)
         canvas.width = img.width;
         canvas.height = chunkSize;
 
         for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
           const datasetBytesView = new Float32Array(
               datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4 *NUM_CHANNELS,
               IMAGE_SIZE * chunkSize*NUM_CHANNELS);
           ctx.drawImage(
               img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
               chunkSize);
 
           const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
           let x = 0
           for (let j = 0; j < imageData.data.length ; j+=4){
             // All channels hold an equal value since the image is grayscale, so
             // just read the red channel.
             for (let i = 0; i < NUM_CHANNELS; i++) {
              datasetBytesView[x++] = imageData.data[j + i] / 255
            }
          }
        }
         this.datasetImages = new Float32Array(datasetBytesBuffer);
 
         resolve();
       }
       img.src = MNIST_IMAGES_SPRITE_PATH;
     });
 
     const labelsRequest = fetch(MNIST_LABELS_PATH);
     const [imgResponse, labelsResponse] =
         await Promise.all([imgRequest, labelsRequest]);
 
     this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
 
     // Create shuffled indices into the train/test set for when we select a
     // random dataset element for training / validation.
     this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
     this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);
 
     // Slice the the images and labels into train and test sets.
     this.trainImages =
         this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS * NUM_CHANNELS);
     this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TEST_ELEMENTS * NUM_CHANNELS);
     this.trainLabels =
         this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
     this.testLabels =
         this.datasetLabels.slice(NUM_CLASSES * NUM_TEST_ELEMENTS);
   }
   
   nextTrainBatch(batchSize) {
     return this.nextBatch(
         batchSize, [this.trainImages, this.trainLabels], () => {
           this.shuffledTrainIndex =
               (this.shuffledTrainIndex + 1) % this.trainIndices.length;
           return this.trainIndices[this.shuffledTrainIndex];
         });
    
   }
 
   nextTestBatch(batchSize) {
     return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
       this.shuffledTestIndex =
           (this.shuffledTestIndex + 1) % this.testIndices.length;
       return this.testIndices[this.shuffledTestIndex];
     });
   }
 
   nextBatch(batchSize, data, index) {
     const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE * NUM_CHANNELS);
     const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);
 
     for (let i = 0; i < batchSize; i++) {
       const idx = index();
       const startPoint = idx * IMAGE_SIZE * NUM_CHANNELS
       const image =
           data[0].slice(startPoint, startPoint + IMAGE_SIZE * NUM_CHANNELS);
       batchImagesArray.set(image, i * IMAGE_SIZE * NUM_CHANNELS);
 
       const label =
           data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
       batchLabelsArray.set(label, i * NUM_CLASSES);
     }
 
     const xs = tf.tensor3d(batchImagesArray, [batchSize, IMAGE_SIZE,NUM_CHANNELS]);
     const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);
     console.log(xs[0],labels[0])
     return {xs, labels};
   }
 }
 