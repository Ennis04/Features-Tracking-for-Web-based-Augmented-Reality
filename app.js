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
const backBtn = document.getElementById("back-btn");

let isGenerated = false;
let uploadedImageData = "";
let selectedModel = "";
let originalImage = new Image();
let webcamStream = null;

// Upload image
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

    originalImage.onload = function () {
      showOriginalPreview();
      actionButtons.classList.remove("hidden");

      isGenerated = false;
      selectedModel = "";
      modelPanel.classList.add("hidden");
      disableActionButtons();
      disableStartArButton();

      statusMessage.textContent =
        "Image uploaded successfully. The original image is shown in preview. Click Generate to perform feature analysis.";
    };

    originalImage.src = uploadedImageData;
  };

  reader.readAsDataURL(file);
});

// Generate feature analysis
generateBtn.addEventListener("click", function () {
  if (!uploadedImageData) {
    statusMessage.textContent = "Please upload an image first.";
    return;
  }

  performOrbStyleFeatureDetection(originalImage);

  isGenerated = true;
  enableActionButtons();
  statusMessage.textContent =
    "Feature analysis completed successfully. The grayscale image and detected feature points are shown in preview.";
});

// Open dropdown model selection
linkModelBtn.addEventListener("click", function () {
  if (!isGenerated) {
    statusMessage.textContent = "Please click Generate before selecting a 3D model.";
    return;
  }

  modelPanel.classList.remove("hidden");
  statusMessage.textContent =
    "Please choose one of the predefined 3D model options.";
});

// Dropdown selection only
modelSelect.addEventListener("change", function () {
  selectedModel = this.value;

  if (!selectedModel) {
    disableStartArButton();
    statusMessage.textContent = "No 3D model selected.";
    return;
  }

  enableStartArButton();
  statusMessage.textContent = `3D model selected: ${selectedModel}. Start AR is now enabled.`;
});

// Start AR interaction
startArBtn.addEventListener("click", async function () {
  if (startArBtn.disabled) {
    return;
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    arVideo.srcObject = webcamStream;
    arView.classList.remove("hidden");
    document.body.style.overflow = "hidden";

    statusMessage.textContent = `AR view started with model: ${selectedModel}.`;
  } catch (error) {
    statusMessage.textContent = "Unable to access the webcam. Please allow camera permission and try again.";
  }

  backBtn.addEventListener("click", function () {
    closeArView();
  });
});

// Save generated result
saveBtn.addEventListener("click", function () {
  if (!isGenerated) {
    statusMessage.textContent = "Save is disabled until Generate is clicked.";
    return;
  }

  const link = document.createElement("a");
  link.href = previewCanvas.toDataURL("image/png");
  link.download = "feature-analysis-result.png";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  statusMessage.textContent = "Generated image saved to your local device.";
});

function showOriginalPreview() {
  previewPlaceholder.style.display = "none";
  previewCanvas.style.display = "none";
  imagePreview.style.display = "block";
  imagePreview.src = uploadedImageData;
}

function showCanvasPreview() {
  previewPlaceholder.style.display = "none";
  imagePreview.style.display = "none";
  previewCanvas.style.display = "block";
}

function disableActionButtons() {
  saveBtn.disabled = true;
  saveBtn.classList.add("btn-disabled");

  linkModelBtn.disabled = true;
  linkModelBtn.classList.add("btn-disabled");
}

function enableActionButtons() {
  saveBtn.disabled = false;
  saveBtn.classList.remove("btn-disabled");

  linkModelBtn.disabled = false;
  linkModelBtn.classList.remove("btn-disabled");
}

function disableStartArButton() {
  startArBtn.disabled = true;
  startArBtn.classList.add("btn-disabled");
}

function enableStartArButton() {
  startArBtn.disabled = false;
  startArBtn.classList.remove("btn-disabled");
}

function closeArView() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }

  arVideo.srcObject = null;
  arView.classList.add("hidden");
  document.body.style.overflow = "";
}

function resetPreview() {
  closeArView();

  imagePreview.src = "";
  imagePreview.style.display = "none";
  previewCanvas.style.display = "none";
  previewPlaceholder.style.display = "block";

  fileNameText.textContent = "No file selected";
  actionButtons.classList.add("hidden");
  statusMessage.textContent = "";
  uploadedImageData = "";
  isGenerated = false;
  selectedModel = "";

  if (modelSelect) {
    modelSelect.value = "";
  }

  modelPanel.classList.add("hidden");

  disableActionButtons();
  disableStartArButton();
}

function performOrbStyleFeatureDetection(img) {
  const ctx = previewCanvas.getContext("2d");

  const maxWidth = 700;
  let width = img.width;
  let height = img.height;

  if (width > maxWidth) {
    const scale = maxWidth / width;
    width = Math.floor(width * scale);
    height = Math.floor(height * scale);
  }

  previewCanvas.width = width;
  previewCanvas.height = height;

  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(img, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  const gray = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

      const grayValue = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      gray[y * width + x] = grayValue;

      data[i] = grayValue;
      data[i + 1] = grayValue;
      data[i + 2] = grayValue;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  const circleOffsets = [
    [0, -3], [1, -3], [2, -2], [3, -1],
    [3, 0], [3, 1], [2, 2], [1, 3],
    [0, 3], [-1, 3], [-2, 2], [-3, 1],
    [-3, 0], [-3, -1], [-2, -2], [-1, -3]
  ];

  const threshold = 30;
  const minArcLength = 9;
  const keypoints = [];

  for (let y = 4; y < height - 4; y++) {
    for (let x = 4; x < width - 4; x++) {
      const center = gray[y * width + x];
      const brighter = [];
      const darker = [];

      for (let k = 0; k < 16; k++) {
        const nx = x + circleOffsets[k][0];
        const ny = y + circleOffsets[k][1];
        const value = gray[ny * width + nx];

        brighter.push(value >= center + threshold ? 1 : 0);
        darker.push(value <= center - threshold ? 1 : 0);
      }

      const brighterExtended = brighter.concat(brighter);
      const darkerExtended = darker.concat(darker);

      let isCorner = false;
      let score = 0;

      for (let start = 0; start < 16; start++) {
        let brightCount = 0;
        let darkCount = 0;

        for (let j = 0; j < minArcLength; j++) {
          brightCount += brighterExtended[start + j];
          darkCount += darkerExtended[start + j];
        }

        if (brightCount === minArcLength || darkCount === minArcLength) {
          isCorner = true;

          for (let k = 0; k < 16; k++) {
            const nx = x + circleOffsets[k][0];
            const ny = y + circleOffsets[k][1];
            const value = gray[ny * width + nx];
            score += Math.abs(value - center);
          }

          break;
        }
      }

      if (isCorner) {
        keypoints.push({ x, y, score });
      }
    }
  }

  const suppressed = [];
  const radius = 10;

  keypoints.sort((a, b) => b.score - a.score);

  for (const kp of keypoints) {
    let keep = true;

    for (const chosen of suppressed) {
      const dx = kp.x - chosen.x;
      const dy = kp.y - chosen.y;
      if (Math.sqrt(dx * dx + dy * dy) < radius) {
        keep = false;
        break;
      }
    }

    if (keep) {
      suppressed.push(kp);
    }

    if (suppressed.length >= 150) {
      break;
    }
  }

  const orientedKeypoints = suppressed.map((kp) => {
    let m01 = 0;
    let m10 = 0;
    const patchRadius = 8;

    for (let py = -patchRadius; py <= patchRadius; py++) {
      for (let px = -patchRadius; px <= patchRadius; px++) {
        const xx = kp.x + px;
        const yy = kp.y + py;

        if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;

        const intensity = gray[yy * width + xx];
        m10 += px * intensity;
        m01 += py * intensity;
      }
    }

    const angle = Math.atan2(m01, m10);

    return { ...kp, angle };
  });

  ctx.fillStyle = "red";
  ctx.strokeStyle = "yellow";
  ctx.lineWidth = 1.5;

  orientedKeypoints.forEach((kp) => {
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 3.5, 0, Math.PI * 2);
    ctx.fill();

    const lineLength = 10;
    const endX = kp.x + Math.cos(kp.angle) * lineLength;
    const endY = kp.y + Math.sin(kp.angle) * lineLength;

    ctx.beginPath();
    ctx.moveTo(kp.x, kp.y);
    ctx.lineTo(endX, endY);
    ctx.stroke();
  });

  showCanvasPreview();
}