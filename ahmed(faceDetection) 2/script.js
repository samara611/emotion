const video = document.getElementById("video");
const container = document.getElementById("container");
const sad=document.getElementById("sad")
const angry=document.getElementById("angry");
const ok=document.getElementById("ok")
const question=document.getElementById("question")
const holder=document.getElementById("Qsholder");
//we call the weights ( the trained model ) with promise and then open the camera
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
  faceapi.nets.faceExpressionNet.loadFromUri("./models"),
  faceapi.nets.ageGenderNet.loadFromUri("./models"),
]).then(webCam);

//the function make to open the camera (without audio) and then stream to the web page 
function webCam() {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((error) => {
      console.log(error);
    });
}
let det

//here we start to detect the face, the expression and the gender (canva)
video.addEventListener("play", () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  container.append(canvas);

  //this line make the blue rect to fit in video dimention
  faceapi.matchDimensions(canvas, { height: video.height, width: video.width });

  setInterval(async () => {
    const detection = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceExpressions().withAgeAndGender();

    sad.innerHTML=`${detection[0].expressions.sad}` // return percentage of sadness
    angry.innerHTML=`${detection[0].expressions.angry}` // return percentage of anger
    checkIfAnger(detection[0].expressions.angry);
    checkIfSad(detection[0].expressions.sad)


    //make the blue rect in the middel of the face in camera dimention
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    //and this function to center the detedtion in the middel of the face
    const resizedWindow = faceapi.resizeResults(detection, {
      height: video.height,
      width: video.width,
    });
    //and thin we start to drow with draw fun in face-api.js
    faceapi.draw.drawDetections(canvas, resizedWindow);
    faceapi.draw.drawFaceExpressions(canvas, resizedWindow);

    //the gender detection has no function in face-api.js and we should to declare this to canvas in line 29
    resizedWindow.forEach((detection) => {
      const box = detection.detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: detection.gender+','+Math.floor(detection.age),
      });
      drawBox.draw(canvas);
    });
    console.log(detection);

  }, 100);
});

function checkIfSad(sadPercentage){
  if(sadPercentage>0.70){
    holder.style.display='flex'
      question.innerHTML="Sadness is detected * link* if you want to change your mood"
      ok.href='https://www.youtube.com/results?search_query=funny+videos'
    setTimeout(()=>{
      reset()
    },5000)
  }
}
function checkIfAnger(angerPercentage){
  if(angerPercentage>0.70){
    holder.style.display='flex'
      question.innerHTML="Anger is detected * link* if you want to change your mood"
      ok.href="https://www.youtube.com/results?search_query=funny+videos"
    setTimeout(()=>{
     reset() 
    },5000)
  }
}
function reset(){
  holder.style.display='none'
  question.innerHTML=''
  ok.href='#'
}