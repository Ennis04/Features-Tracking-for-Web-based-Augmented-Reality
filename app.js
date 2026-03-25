const imageUpload = document.getElementById("image-upload");
const imagePreview = document.getElementById("image-preview");
const previewCanvas = document.getElementById("preview-canvas");
const previewPlaceholder = document.getElementById("preview-placeholder");
const fileNameText = document.getElementById("file-name");
const actionButtons = document.getElementById("action-buttons");
const generateBtn = document.getElementById("generate-btn");
const saveBtn = document.getElementById("save-btn");
const linkModelBtn = document.getElementById("link-model-btn");
const startArBtn = document.getElementById("start-ar-btn");
const statusMessage = document.getElementById("status-message");

const modelPanel = document.getElementById("model-panel");
const modelSelect = document.getElementById("model-select");

const arView = document.getElementById("ar-view");
const arVideo = document.getElementById("ar-video");
const arOverlayCanvas = document.getElementById("ar-overlay-canvas");
const backBtn = document.getElementById("back-btn");

let isGenerated = false;
let uploadedImageData = "";
let selectedModel = "";
let originalImage = new Image();
let webcamStream = null;
let cvReady = false;

let refKeypoints = null;
let refDescriptors = null;
let refWidth = 0;
let refHeight = 0;

let arLoopId = null;
let arCap = null;
let frameGray = null;
let frameSrc = null;
let frameKeypoints = null;
let frameDescriptors = null;
let orb = null;
let bfMatcher = null;

function onOpenCvReady() {
  cv['onRuntimeInitialized'] = () => {
    cvReady = true;
    if (statusMessage.textContent === "" || statusMessage.textContent.includes("wait")) {
      statusMessage.textContent = "OpenCV.js loaded successfully. Ready.";
    }
  };
}

imageUpload.addEventListener("change", function () {
  const file = this.files[0];
  if (!file) { resetPreview(); return; }
  
  fileNameText.textContent = `Selected file: ${file.name}`;
  const reader = new FileReader();
  
  reader.onload = function (event) {
    uploadedImageData = event.target.result;
    originalImage.onload = function () {
      showOriginalPreview();
      actionButtons.classList.remove("hidden");
      isGenerated = false;
      selectedModel = "";
      modelPanel.classList.add("hidden");
      disableActionButtons();
      disableStartArButton();
      statusMessage.textContent = "Image uploaded. Click Generate to perform feature analysis using OpenCV.";
    };
    originalImage.src = uploadedImageData;
  };
  reader.readAsDataURL(file);
});

generateBtn.addEventListener("click", function () {
  if (!uploadedImageData) return;
  if (!cvReady) { statusMessage.textContent = "OpenCV.js loading..."; return; }

  statusMessage.textContent = "Processing image with OpenCV...";

  previewPlaceholder.style.display = "none";
  imagePreview.style.display = "none";
  previewCanvas.style.display = "block";

  const ctx = previewCanvas.getContext("2d", { willReadFrequently: true });
  const maxWidth = 700;
  refWidth = originalImage.width;
  refHeight = originalImage.height;

  if (refWidth > maxWidth) {
    const scale = maxWidth / refWidth;
    refWidth = Math.floor(refWidth * scale);
    refHeight = Math.floor(refHeight * scale);
  }
  
  previewCanvas.width = refWidth;
  previewCanvas.height = refHeight;
  ctx.clearRect(0, 0, refWidth, refHeight);
  ctx.drawImage(originalImage, 0, 0, refWidth, refHeight);

  if (refKeypoints) refKeypoints.delete();
  if (refDescriptors) refDescriptors.delete();

  let src = cv.imread(previewCanvas);
  let gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  //ORB
  if(!orb) orb = new cv.ORB(500); 
  refKeypoints = new cv.KeyPointVector();
  refDescriptors = new cv.Mat();
  
  orb.detectAndCompute(gray, new cv.Mat(), refKeypoints, refDescriptors);

  let outImg = new cv.Mat();
  let dotColor = new cv.Scalar(255, 0, 0, 255); 
  cv.drawKeypoints(gray, refKeypoints, outImg, dotColor);
  cv.imshow("preview-canvas", outImg);

  src.delete(); 
  gray.delete(); 
  outImg.delete();

  isGenerated = true;
  enableActionButtons();
  statusMessage.textContent = `OpenCV Analysis complete! Found ${refKeypoints.size()} ORB features.`;
});

linkModelBtn.addEventListener("click", function () {
  if (!isGenerated) return;
  modelPanel.classList.remove("hidden");
  statusMessage.textContent = "Please choose a 3D model.";
});

modelSelect.addEventListener("change", function () {
  selectedModel = this.value;
  if (!selectedModel) {
    disableStartArButton();
    return;
  }
  enableStartArButton();
  statusMessage.textContent = `3D model selected: ${selectedModel}. Start AR is now enabled.`;
});

startArBtn.addEventListener("click", async function () {
  if (startArBtn.disabled) return;
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    
    arVideo.srcObject = webcamStream;
    
    arVideo.onloadedmetadata = () => {
      arVideo.play();
      
      const checkVideoSize = setInterval(() => {
        if (arVideo.videoWidth > 0) {
          clearInterval(checkVideoSize);
          
          arOverlayCanvas.width = arVideo.videoWidth;
          arOverlayCanvas.height = arVideo.videoHeight;
          
          arCap = new cv.VideoCapture(arVideo);
          frameSrc = new cv.Mat(arVideo.videoHeight, arVideo.videoWidth, cv.CV_8UC4);
          frameGray = new cv.Mat();
          frameKeypoints = new cv.KeyPointVector();
          frameDescriptors = new cv.Mat();
          
          if(!bfMatcher) bfMatcher = new cv.BFMatcher(cv.NORM_HAMMING, true);

          arLoopId = requestAnimationFrame(processAR);
        }
      }, 50);
    };

    arView.classList.remove("hidden");
    document.body.style.overflow = "hidden";
    statusMessage.textContent = `Tracking started. Point camera at the marker.`;
    
  } catch (error) {
    statusMessage.textContent = "Unable to access the webcam.";
  }

  backBtn.addEventListener("click", function () {
    closeArView();
  });
});

function processAR() {
  if (!webcamStream || !cvReady || !arCap) return;
  
  const ctx = arOverlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, arOverlayCanvas.width, arOverlayCanvas.height);
  
  try {
    arCap.read(frameSrc);
    cv.cvtColor(frameSrc, frameGray, cv.COLOR_RGBA2GRAY);

    orb.detectAndCompute(frameGray, new cv.Mat(), frameKeypoints, frameDescriptors);

    if (frameDescriptors.rows > 0 && refDescriptors.rows > 0) {
      
      let matches = new cv.DMatchVector();
      bfMatcher.match(refDescriptors, frameDescriptors, matches);

      let goodMatches = [];
      for (let i = 0; i < matches.size(); i++) {
        let match = matches.get(i);
        if (match.distance < 75) {  
          goodMatches.push(match);
        }
      }

      ctx.fillStyle = "lime";
      for (let i = 0; i < goodMatches.length; i++) {
        let trainIdx = goodMatches[i].trainIdx;
        let pt = frameKeypoints.get(trainIdx).pt;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
      
      if (goodMatches.length >= 8) {
        let srcPts = [];
        let dstPts = [];
        
        for (let i = 0; i < goodMatches.length; i++) {
          let refPt = refKeypoints.get(goodMatches[i].queryIdx).pt;
          let framePt = frameKeypoints.get(goodMatches[i].trainIdx).pt;
          srcPts.push(refPt.x, refPt.y);
          dstPts.push(framePt.x, framePt.y);
        }

        let refMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, srcPts);
        let frameMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, dstPts);

        let H = cv.findHomography(refMat, frameMat, cv.RANSAC, 8.0);
        
        if (!H.empty()) {
          let objCorners = cv.matFromArray(4, 1, cv.CV_32FC2, [
            0, 0, 
            refWidth, 0, 
            refWidth, refHeight, 
            0, refHeight
          ]);
          let sceneCorners = new cv.Mat();
          
          cv.perspectiveTransform(objCorners, sceneCorners, H);

          ctx.strokeStyle = "springgreen";
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.moveTo(sceneCorners.data32F[0], sceneCorners.data32F[1]);
          ctx.lineTo(sceneCorners.data32F[2], sceneCorners.data32F[3]);
          ctx.lineTo(sceneCorners.data32F[4], sceneCorners.data32F[5]);
          ctx.lineTo(sceneCorners.data32F[6], sceneCorners.data32F[7]);
          ctx.closePath();
          ctx.stroke();

          ctx.fillStyle = "springgreen";
          ctx.font = "bold 24px Arial";
          ctx.fillText("TRACKED", sceneCorners.data32F[0], sceneCorners.data32F[1] - 10);

          objCorners.delete();
          sceneCorners.delete();
        }
        
        refMat.delete();
        frameMat.delete();
        H.delete();
      }
      matches.delete();
    }
  } catch(e) {
    console.error(e);
  }

  arLoopId = requestAnimationFrame(processAR);
}

saveBtn.addEventListener("click", function () {
  if (!isGenerated) return;
  const link = document.createElement("a");
  link.href = previewCanvas.toDataURL("image/png");
  link.download = "feature-analysis-result.png";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

function showOriginalPreview() {
  previewPlaceholder.style.display = "none";
  previewCanvas.style.display = "none";
  imagePreview.style.display = "block";
  imagePreview.src = uploadedImageData;
}

function disableActionButtons() {
  saveBtn.disabled = true; saveBtn.classList.add("btn-disabled");
  linkModelBtn.disabled = true; linkModelBtn.classList.add("btn-disabled");
}
function enableActionButtons() {
  saveBtn.disabled = false; saveBtn.classList.remove("btn-disabled");
  linkModelBtn.disabled = false; linkModelBtn.classList.remove("btn-disabled");
}
function disableStartArButton() { startArBtn.disabled = true; startArBtn.classList.add("btn-disabled"); }
function enableStartArButton() { startArBtn.disabled = false; startArBtn.classList.remove("btn-disabled"); }

function closeArView() {
  if (arLoopId) { cancelAnimationFrame(arLoopId); arLoopId = null; }
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }
  
  if (frameSrc) { frameSrc.delete(); frameSrc = null; }
  if (frameGray) { frameGray.delete(); frameGray = null; }
  if (frameKeypoints) { frameKeypoints.delete(); frameKeypoints = null; }
  if (frameDescriptors) { frameDescriptors.delete(); frameDescriptors = null; }

  arVideo.srcObject = null;
  arView.classList.add("hidden");
  document.body.style.overflow = "";
}

function resetPreview() {
  closeArView();
  imagePreview.src = ""; imagePreview.style.display = "none";
  previewCanvas.style.display = "none";
  previewPlaceholder.style.display = "block";
  fileNameText.textContent = "No file selected";
  actionButtons.classList.add("hidden");
  statusMessage.textContent = "";

  uploadedImageData = ""; isGenerated = false; selectedModel = "";
  if (modelSelect) modelSelect.value = "";
  modelPanel.classList.add("hidden");
  disableActionButtons(); disableStartArButton();
}