
function displayRandomImages()   
{  
   var imageNameArray = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];
      
    // //find the length of the array of images  
    // var arrayLength = imageArray.length;  
    // var newArray = [];  
    // for (var i = 0; i < arrayLength; i++) {  
    //     newArray[i] = new Image();  
    //     newArray[i].src = imageArray[i].src;  
    //     newArray[i].width = imageArray[i].width;  
    //     newArray[i].height = imageArray[i].height;  
    // }  
     
  // create random image number  
  function getRandomNum(min, max)   
  {  
      // generate and return a random number for the image to be displayed   
      imgNo = Math.floor(Math.random() * (max - min + 1)) + min;  
      return imageNameArray[imgNo];  
  }    
  
  // 0 is first image and (preBuffer.length - 1) is last image of the array  
  var randomLetter = getRandomNum(0, imageNameArray.length - 1); 
  var img_url = '{% static "/data/pics/filename.png" %}'.replace(filename,randomLetter);
  var newimg = document.createElement("newimg");
  newimg.src = img_url;
   
  // remove the previous images  
  var images = document.getElementsByTagName('letterImg');  
  var l = images.length;  
  for (var p = 0; p < l; p++) {  
     images[0].parentNode.removeChild(images[0]);  
  }  
  // display the new random image    
  document.body.appendChild(newimg);  
}  

  
