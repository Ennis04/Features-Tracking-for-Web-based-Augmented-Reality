const imageUpload = document.getElementById("image-upload");
const imagePreview = document.getElementById("image-preview");
const previewPlaceholder = document.getElementById("preview-placeholder");
const fileNameText = document.getElementById("file-name");
const actionButtons = document.getElementById("action-buttons");
const generateBtn = document.getElementById("generate-btn");
const saveBtn = document.getElementById("save-btn");
const statusMessage = document.getElementById("status-message");

let isGenerated = false;
let uploadedImageData = "";
let cvReady = false;

function onOpenCvReady() {
  // OpenCV.js uses WebAssembly which initializes asynchronously.
  cv['onRuntimeInitialized'] = () => {
    cvReady = true;
    const statusMsg = document.getElementById("status-message");
    if (statusMsg.textContent === "" || statusMsg.textContent.includes("wait")) {
      statusMsg.textContent = "OpenCV.js loaded successfully. Ready.";
    }
  };
}

// Handle image upload
imageUpload.addEventListener("change", function () {
  const file = this.files[0];

  if (!file) {
    resetPreview();
    return;
  }

  fileNameText.textContent = `Selected file: ${file.name}`;

  const reader = new FileReader();

  reader.onload = function (event) {
    uploadedImageData = event.target.result;
    imagePreview.src = uploadedImageData;
    imagePreview.style.display = "block";
    previewPlaceholder.style.display = "none";
    document.getElementById("output-canvas").style.display = "none";

    actionButtons.classList.remove("hidden");

    // Reset state whenever a new image is uploaded
    isGenerated = false;
    disableSaveButton();
    statusMessage.textContent = "Image uploaded successfully. Click Generate to generate features.";
  };

  reader.readAsDataURL(file);
});

// Generate button logic
generateBtn.addEventListener("click", function () {
  if (!uploadedImageData) {
    statusMessage.textContent = "Please upload an image first.";
    return;
  }
  if (!cvReady) {
    statusMessage.textContent = "OpenCV.js not loaded.";
    return;
  }

  try {
    statusMessage.textContent = "Processing image...";

    const srcImg = document.getElementById("image-preview");
    const src = cv.imread(srcImg);
    
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // ORB
    const orb = new cv.ORB(500); // Max 500 features
    const keypoints = new cv.KeyPointVector();
    const descriptors = new cv.Mat();
    
    orb.detect(gray, keypoints);

    const outImg = new cv.Mat();
    const color = new cv.Scalar(255, 0, 0, 255);
    cv.drawKeypoints(gray, keypoints, outImg, color);
    
    const outputCanvas = document.getElementById("output-canvas");
    cv.imshow("output-canvas", outImg);
    
    srcImg.style.display = "none";
    outputCanvas.style.display = "block";

    const numFeatures = keypoints.size();

    // Clean up memory
    src.delete();
    gray.delete();
    orb.delete();
    keypoints.delete();
    descriptors.delete();
    outImg.delete();

    isGenerated = true;
    enableSaveButton();
    statusMessage.textContent = `Features generated successfully! Found ${numFeatures} ORB features on grayscale image.`;
  } catch (err) {
    console.error("OpenCV Processing Error: ", err);
    statusMessage.textContent = "Error generating features: " + err.message;
  }
});

// Save button logic
saveBtn.addEventListener("click", function () {
  if (!isGenerated) {
    statusMessage.textContent = "Save is disabled until Generate is clicked.";
    return;
  }

  // Placeholder save response
  statusMessage.textContent = "Save function is ready to be implemented.";
});

function disableSaveButton() {
  saveBtn.disabled = true;
  saveBtn.classList.add("btn-disabled");
}

function enableSaveButton() {
  saveBtn.disabled = false;
  saveBtn.classList.remove("btn-disabled");
}

function resetPreview() {
  imagePreview.src = "";
  imagePreview.style.display = "none";
  const outputCanvas = document.getElementById("output-canvas");
  if (outputCanvas) outputCanvas.style.display = "none";
  previewPlaceholder.style.display = "block";
  fileNameText.textContent = "No file selected";
  actionButtons.classList.add("hidden");
  statusMessage.textContent = "";
  uploadedImageData = "";
  isGenerated = false;
  disableSaveButton();
}