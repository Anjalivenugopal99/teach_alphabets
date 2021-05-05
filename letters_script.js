import {LMnistData} from './letters_data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;

tf.loadLayersModel('model/my_model.json').then(function(model) {
    window.model = model;
  });


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

function getAndAddImage(){
    var imageNameArray = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];
    // 0 is first image and (preBuffer.length - 1) is last image of the array  
    function getRandomNum(min, max){  
        // generate and return a random number for the image to be displayed   
        var imgNo = Math.floor(Math.random() * (max - min + 1)) + min;  
        return imageNameArray[imgNo];  
    }  
    var randomLetter = getRandomNum(0, imageNameArray.length - 1);

    // var img_url = '{% static "/data/pics/filename.png" %}'.replace(filename,randomLetter);
    var img_url = "./data/pics/A.png".replace('A',randomLetter);
    // var newimg = document.createElement("newimg");
    // newimg.src = img_url;

    
    // remove the previous images  
    var images = document.getElementById('letterImg'); 
    images.src = img_url;
    images.setAttribute("letter",randomLetter);
    // var l = images.length;  
    // for (var p = 0; p < l; p++) {  
    //     images[0].parentNode.removeChild(images[0]);  
    // }  
    // // display the new random image    
    // document.getElementById('letterImg').appendChild(newimg); 

}
    
function save() {
    var raw = tf.browser.fromPixels(rawImage,1);
    var resized = tf.image.resizeBilinear(raw, [28,28]);
    var tensor = resized.expandDims(0);
    
    var prediction = window.model.predict(tensor);
    console.log("prediction"+ prediction);
    var pIndex = tf.argMax(prediction, 1).dataSync();
    console.log("pIndex"+ pIndex);
    
    var classNames = ["A", "B", "C", 
                      "D", "E", "F", "G",
                      "H", "I", "J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];
            
            
    var predictedLetter = classNames[pIndex];
    var images = document.getElementById('letterImg'); 
    var randomLetter = images.getAttribute("letter");
    if(predictedLetter == randomLetter)
    {
        alert("You are correct!");
        getAndAddImage();
        erase();

    }
    else{
        alert("Oops!! Try again")
    }


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
    getAndAddImage();
}


async function run() {
    
    
    init();

}

document.addEventListener('DOMContentLoaded', run);
