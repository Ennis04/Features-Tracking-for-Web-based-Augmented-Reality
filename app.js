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

const MATCH_DISTANCE_THRESHOLD = 16;
const MATCH_RATIO_THRESHOLD = 0.82;
const MIN_MATCH_COUNT = 8;
const MIN_INLIER_COUNT = 5;
const RANSAC_INLIER_THRESHOLD = 14;
const RANSAC_ITERATIONS = 250;

const arOverlay = document.getElementById("ar-overlay");
const SHOW_ALL_FRAME_KEYPOINTS = true;

const FRAME_PROCESS_INTERVAL = 120; // ms, about 8 feature analyses per second
const FRAME_MAX_WIDTH = 480;

const frameCanvas = document.createElement("canvas");
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });

const threeCanvas = document.getElementById("three-canvas");

let THREE_MODULE = null;
let GLTFLoaderClass = null;

let isGenerated = false;
let uploadedImageData = "";
let selectedModel = "";
let originalImage = new Image();
let webcamStream = null;
let targetFeatures = null;
let arLoopId = null;
let isArRunning = false;
let lastFrameProcessTime = 0;
let latestFrameFeatures = null;
let latestTrackingResult = null;

let renderer = null;
let scene = null;
let arCamera3D = null;
let ambientLight = null;
let directionalLight = null;

let smoothedPosition = null;
let smoothedScale = 1;
let smoothedRotationZ = 0;

let trackedObjectRoot = null;
let activeModel = null;
let gltfLoader = null;

const DESCRIPTOR_PAIRS = [
  { ax: -4, ay: -4, bx: 4, by: 4 },
  { ax: -4, ay: 4, bx: 4, by: -4 },
  { ax: -5, ay: 0, bx: 5, by: 0 },
  { ax: 0, ay: -5, bx: 0, by: 5 },
  { ax: -3, ay: -1, bx: 3, by: 1 },
  { ax: -3, ay: 1, bx: 3, by: -1 },
  { ax: -2, ay: -4, bx: 2, by: 4 },
  { ax: -2, ay: 4, bx: 2, by: -4 },
  { ax: -6, ay: -2, bx: 2, by: 3 },
  { ax: -2, ay: -6, bx: 3, by: 2 },
  { ax: -1, ay: -5, bx: 1, by: 5 },
  { ax: -5, ay: -1, bx: 5, by: 1 },
  { ax: -6, ay: 1, bx: 4, by: -2 },
  { ax: -4, ay: 2, bx: 6, by: -1 },
  { ax: -3, ay: -5, bx: 3, by: 5 },
  { ax: -5, ay: -3, bx: 5, by: 3 },
  { ax: -1, ay: -3, bx: 1, by: 3 },
  { ax: -3, ay: -1, bx: 3, by: 1 },
  { ax: -6, ay: 0, bx: 6, by: 0 },
  { ax: 0, ay: -6, bx: 0, by: 6 },
  { ax: -4, ay: -2, bx: 4, by: 2 },
  { ax: -4, ay: 2, bx: 4, by: -2 },
  { ax: -2, ay: -2, bx: 2, by: 2 },
  { ax: -2, ay: 2, bx: 2, by: -2 },
  { ax: -7, ay: 0, bx: 3, by: 1 },
  { ax: -3, ay: 1, bx: 7, by: 0 },
  { ax: -1, ay: -7, bx: 1, by: 3 },
  { ax: -1, ay: 3, bx: 1, by: 7 },
  { ax: -7, ay: -2, bx: 2, by: 5 },
  { ax: -5, ay: 2, bx: 7, by: -2 },
  { ax: -2, ay: -7, bx: 5, by: 2 },
  { ax: -5, ay: -2, bx: 2, by: 7 },
  { ax: -6, ay: -4, bx: 4, by: 6 },
  { ax: -4, ay: 6, bx: 6, by: -4 },
  { ax: -6, ay: 4, bx: 4, by: -6 },
  { ax: -4, ay: -6, bx: 6, by: 4 },
  { ax: -3, ay: -6, bx: 3, by: 6 },
  { ax: -6, ay: -3, bx: 6, by: 3 },
  { ax: -7, ay: 1, bx: 5, by: -1 },
  { ax: -5, ay: 1, bx: 7, by: -1 },
  { ax: -1, ay: -7, bx: 5, by: 1 },
  { ax: -5, ay: -1, bx: 1, by: 7 },
  { ax: -3, ay: 0, bx: 3, by: 0 },
  { ax: 0, ay: -3, bx: 0, by: 3 },
  { ax: -7, ay: -3, bx: 1, by: 6 },
  { ax: -1, ay: -6, bx: 7, by: 3 },
  { ax: -6, ay: 1, bx: 3, by: -7 },
  { ax: -3, ay: 7, bx: 6, by: -1 },
  { ax: -8, ay: 0, bx: 0, by: 4 },
  { ax: 0, ay: -8, bx: 4, by: 0 },
  { ax: -4, ay: 0, bx: 8, by: 0 },
  { ax: 0, ay: -4, bx: 0, by: 8 },
  { ax: -7, ay: 2, bx: 6, by: -2 },
  { ax: -6, ay: 2, bx: 7, by: -2 },
  { ax: -2, ay: -7, bx: 2, by: 6 },
  { ax: -2, ay: -6, bx: 2, by: 7 },
  { ax: -8, ay: -2, bx: 4, by: 3 },
  { ax: -4, ay: -3, bx: 8, by: 2 },
  { ax: -3, ay: -8, bx: 3, by: 4 },
  { ax: -3, ay: -4, bx: 3, by: 8 },
  { ax: -5, ay: -5, bx: 5, by: 5 },
  { ax: -5, ay: 5, bx: 5, by: -5 },
  { ax: -8, ay: 1, bx: 2, by: -3 },
  { ax: -2, ay: 3, bx: 8, by: -1 }
];

const MODEL_PATHS = {
  sphere: "model/cube/scene.gltf",
  cube: "model/cube/scene.gltf",
  pyramid: "model/cube/scene.gltf",
  cuboid: "model/cube/scene.gltf"
};

async function ensureThreeLoaded() {
  if (THREE_MODULE && GLTFLoaderClass) {
    return;
  }

  const threeImport = await import("https://esm.sh/three@0.160.0");
  const loaderImport = await import("https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js");

  THREE_MODULE = threeImport;
  GLTFLoaderClass = loaderImport.GLTFLoader;
}

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
      targetFeatures = null;

      modelPanel.classList.add("hidden");
      disableActionButtons();
      disableStartArButton();

      statusMessage.textContent =
        "Image uploaded successfully. The original image is shown in preview. Click Generate to extract target features.";
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

  targetFeatures = extractTargetFeatures(originalImage);

  if (!targetFeatures || targetFeatures.keypoints.length === 0) {
    isGenerated = false;
    disableActionButtons();
    disableStartArButton();
    modelPanel.classList.add("hidden");
    statusMessage.textContent =
      "No stable feature points were found. Try another image with more corners, texture, or contrast.";
    return;
  }

  drawFeaturePreview(targetFeatures);

  isGenerated = true;
  enableActionButtons();

  statusMessage.textContent =
    `Feature analysis completed successfully. ${targetFeatures.keypoints.length} keypoints with descriptors were extracted.`;
});

// Open dropdown model selection
linkModelBtn.addEventListener("click", function () {
  if (!isGenerated || !targetFeatures) {
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
  if (startArBtn.disabled) return;

  if (!targetFeatures || targetFeatures.keypoints.length === 0) {
    statusMessage.textContent = "Please generate target features before starting AR.";
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusMessage.textContent = "This browser does not support webcam access.";
    return;
  }

  try {
    // Step 1: get camera
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false
      });
    } catch (cameraError) {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
    }

    // Step 2: attach stream to video
    arVideo.srcObject = webcamStream;

    // Step 3: show AR view first
    arView.classList.remove("hidden");
    document.body.style.overflow = "hidden";

    // Step 4: wait until video metadata is ready
    await new Promise((resolve, reject) => {
      arVideo.onloadedmetadata = () => resolve();
      arVideo.onerror = () => reject(new Error("Video metadata failed to load"));
    });

    await arVideo.play();

    // Step 5: setup overlay only
    setupArOverlay();

    // Step 6: load Three.js first, then setup 3D
    try {
      await ensureThreeLoaded();
      setupThreeScene();
      resizeThreeScene();
    } catch (threeError) {
      console.error("Three.js setup failed:", threeError);
      statusMessage.textContent = `Camera started, but 3D setup failed: ${threeError.message}`;
      return;
    }

    // Step 7: load model separately
    try {
      if (selectedModel) {
        await loadSelectedModel(selectedModel);
      }
    } catch (modelError) {
      console.error("Model load failed:", modelError);
      statusMessage.textContent = `Camera started, but model loading failed: ${modelError.message}`;
      return;
    }

    // Step 8: start tracking loop
    startArProcessingLoop();

    statusMessage.textContent = `AR view started with model: ${selectedModel}.`;
  } catch (error) {
    console.error("Start AR failed:", error);
    statusMessage.textContent = `Start AR failed: ${error.name || "Error"} - ${error.message || ""}`;
  }
});

// Attach once only
backBtn.addEventListener("click", function () {
  closeArView();
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
  stopArProcessingLoop();

  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }

  arVideo.srcObject = null;
  arView.classList.add("hidden");
  document.body.style.overflow = "";

  if (trackedObjectRoot) {
    trackedObjectRoot.visible = false;
  }

  if (renderer) {
    renderer.dispose();
    renderer = null;
  }

  smoothedPosition = null;
  smoothedScale = 1;
  smoothedRotationZ = 0;

  scene = null;
  arCamera3D = null;
  activeModel = null;
  gltfLoader = null;
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
  targetFeatures = null;

  if (modelSelect) {
    modelSelect.value = "";
  }

  modelPanel.classList.add("hidden");

  disableActionButtons();
  disableStartArButton();
}

function setupArOverlay() {
  const rect = arView.getBoundingClientRect();
  arOverlay.width = rect.width;
  arOverlay.height = rect.height;
}

window.addEventListener("resize", function () {
  if (!arView.classList.contains("hidden")) {
    setupArOverlay();
    resizeThreeScene();
  }
});

function startArProcessingLoop() {
  if (isArRunning) {
    return;
  }

  isArRunning = true;
  lastFrameProcessTime = 0;
  latestTrackingResult = null;

  const loop = (timestamp) => {
    if (!isArRunning) {
      return;
    }

    arLoopId = requestAnimationFrame(loop);

    if (!arVideo.videoWidth || !arVideo.videoHeight) {
      return;
    }

    if (timestamp - lastFrameProcessTime < FRAME_PROCESS_INTERVAL) {
      drawArOverlay();
      return;
    }

    lastFrameProcessTime = timestamp;

    latestFrameFeatures = extractFrameFeaturesFromVideo(arVideo);

    if (targetFeatures && latestFrameFeatures) {
      latestTrackingResult = trackTargetInFrame(targetFeatures, latestFrameFeatures);
    } else {
      latestTrackingResult = null;
    }

    drawArOverlay();
    updateTracked3DObject(latestTrackingResult);
    renderThreeScene();
  };

  arLoopId = requestAnimationFrame(loop);
}

function stopArProcessingLoop() {
  isArRunning = false;

  if (arLoopId !== null) {
    cancelAnimationFrame(arLoopId);
    arLoopId = null;
  }

  latestFrameFeatures = null;
  latestTrackingResult = null;

  const ctx = arOverlay.getContext("2d");
  ctx.clearRect(0, 0, arOverlay.width, arOverlay.height);
}

function extractFrameFeaturesFromVideo(video) {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;

  if (!sourceWidth || !sourceHeight) {
    return null;
  }

  let width = sourceWidth;
  let height = sourceHeight;

  if (width > FRAME_MAX_WIDTH) {
    const scale = FRAME_MAX_WIDTH / width;
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }

  frameCanvas.width = width;
  frameCanvas.height = height;

  frameCtx.drawImage(video, 0, 0, width, height);

  const imageData = frameCtx.getImageData(0, 0, width, height);
  const gray = toGrayscale(imageData.data, width, height);
  const blurredGray = blurGray(gray, width, height);

  const rawKeypoints = detectFastLikeCorners(blurredGray, width, height, {
    threshold: 35,
    minArcLength: 9
  });

  const filteredKeypoints = nonMaxSuppression(rawKeypoints, {
    radius: 10,
    maxPoints: 120
  });

  const describedKeypoints = [];

  for (const kp of filteredKeypoints) {
    const angle = estimateOrientation(blurredGray, width, height, kp.x, kp.y, 8);

    const descriptor = computeSimpleBinaryDescriptor(
      blurredGray,
      width,
      height,
      kp.x,
      kp.y,
      angle
    );

    if (!descriptor) {
      continue;
    }

    describedKeypoints.push({
      x: kp.x,
      y: kp.y,
      score: kp.score,
      angle,
      descriptor
    });
  }

  return {
    width,
    height,
    gray: blurredGray,
    keypoints: describedKeypoints,
    timestamp: performance.now()
  };
}

function drawArOverlay() {
  const ctx = arOverlay.getContext("2d");
  ctx.clearRect(0, 0, arOverlay.width, arOverlay.height);

  if (!latestFrameFeatures) return;

  const scaleX = arOverlay.width / latestFrameFeatures.width;
  const scaleY = arOverlay.height / latestFrameFeatures.height;

  if (SHOW_ALL_FRAME_KEYPOINTS) {
    ctx.fillStyle = "rgba(255, 80, 80, 0.9)";
    ctx.strokeStyle = "rgba(255, 230, 80, 0.9)";
    ctx.lineWidth = 1.2;

    for (const kp of latestFrameFeatures.keypoints) {
      const x = kp.x * scaleX;
      const y = kp.y * scaleY;

      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();

      const lineLength = 8;
      const dx = Math.cos(kp.angle) * lineLength;
      const dy = Math.sin(kp.angle) * lineLength;

      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + dx, y + dy);
      ctx.stroke();
    }
  }

  if (!latestTrackingResult) {
    return;
  }

  // Draw inlier matches
  if (latestTrackingResult.inlierMatches.length > 0) {
    ctx.strokeStyle = "rgba(0, 255, 255, 0.9)";
    ctx.lineWidth = 2;

    for (const match of latestTrackingResult.inlierMatches) {
      const x = match.framePoint.x * scaleX;
      const y = match.framePoint.y * scaleY;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // Draw projected target corners if found
  if (latestTrackingResult.found && latestTrackingResult.projectedCorners) {
    const corners = latestTrackingResult.projectedCorners.map((p) => ({
      x: p.x * scaleX,
      y: p.y * scaleY
    }));

    ctx.strokeStyle = "rgba(50, 255, 100, 0.95)";
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < corners.length; i++) {
      ctx.lineTo(corners[i].x, corners[i].y);
    }
    ctx.closePath();
    ctx.stroke();

    // corner markers
    ctx.fillStyle = "rgba(50, 255, 100, 0.95)";
    for (const c of corners) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function updateTrackingStatus(frameFeatures, trackingResult) {
  if (!frameFeatures) {
    return;
  }

  const targetCount = targetFeatures?.keypoints.length || 0;
  const frameCount = frameFeatures.keypoints.length;

  if (!trackingResult) {
    statusMessage.textContent =
      `AR live loop running. Target keypoints: ${targetCount}, frame keypoints: ${frameCount}, matches: 0.`;
    return;
  }

  const rawMatches = trackingResult.rawMatches.length;
  const inliers = trackingResult.inlierMatches.length;

  if (trackingResult.found) {
    statusMessage.textContent =
      `Target FOUND. Target keypoints: ${targetCount}, frame keypoints: ${frameCount}, raw matches: ${rawMatches}, inliers: ${inliers}.`;
  } else {
    statusMessage.textContent =
      `Tracking... Target keypoints: ${targetCount}, frame keypoints: ${frameCount}, raw matches: ${rawMatches}, inliers: ${inliers}.`;
  }
}

// ---------- PART 2: FEATURE EXTRACTION ----------

function extractTargetFeatures(img) {
  const maxWidth = 480;
  let width = img.width;
  let height = img.height;

  if (width > maxWidth) {
    const scale = maxWidth / width;
    width = Math.floor(width * scale);
    height = Math.floor(height * scale);
  }

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = width;
  tempCanvas.height = height;
  const ctx = tempCanvas.getContext("2d");

  ctx.drawImage(img, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const gray = toGrayscale(imageData.data, width, height);
  const blurredGray = blurGray(gray, width, height);

  const rawKeypoints = detectFastLikeCorners(blurredGray, width, height, {
    threshold: 35,
    minArcLength: 9
  });

  const filteredKeypoints = nonMaxSuppression(rawKeypoints, {
    radius: 10,
    maxPoints: 140
  });

  const describedKeypoints = [];

  for (const kp of filteredKeypoints) {
    const angle = estimateOrientation(blurredGray, width, height, kp.x, kp.y, 8);
    const descriptor = computeSimpleBinaryDescriptor(
      blurredGray,
      width,
      height,
      kp.x,
      kp.y,
      angle
    );

    if (!descriptor) {
      continue;
    }

    describedKeypoints.push({
      x: kp.x,
      y: kp.y,
      score: kp.score,
      angle,
      descriptor
    });
  }

  return {
    width,
    height,
    gray: blurredGray,
    keypoints: describedKeypoints
  };
}

function drawFeaturePreview(featureData) {
  const { width, height, gray, keypoints } = featureData;
  const ctx = previewCanvas.getContext("2d");

  previewCanvas.width = width;
  previewCanvas.height = height;

  const imageData = ctx.createImageData(width, height);

  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    const base = i * 4;
    imageData.data[base] = v;
    imageData.data[base + 1] = v;
    imageData.data[base + 2] = v;
    imageData.data[base + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);

  ctx.fillStyle = "red";
  ctx.strokeStyle = "yellow";
  ctx.lineWidth = 1.5;

  for (const kp of keypoints) {
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
    ctx.fill();

    const lineLength = 10;
    const dx = Math.cos(kp.angle) * lineLength;
    const dy = Math.sin(kp.angle) * lineLength;

    ctx.beginPath();
    ctx.moveTo(kp.x, kp.y);
    ctx.lineTo(kp.x + dx, kp.y + dy);
    ctx.stroke();
  }

  showCanvasPreview();
}

function toGrayscale(rgbaData, width, height) {
  const gray = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = rgbaData[i];
      const g = rgbaData[i + 1];
      const b = rgbaData[i + 2];

      gray[y * width + x] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }
  }

  return gray;
}

function blurGray(gray, width, height) {
  const out = new Uint8Array(width * height);

  // Copy borders directly
  for (let x = 0; x < width; x++) {
    out[x] = gray[x];
    out[(height - 1) * width + x] = gray[(height - 1) * width + x];
  }

  for (let y = 0; y < height; y++) {
    out[y * width] = gray[y * width];
    out[y * width + (width - 1)] = gray[y * width + (width - 1)];
  }

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sum = 0;

      sum += gray[(y - 1) * width + (x - 1)];
      sum += gray[(y - 1) * width + x] * 2;
      sum += gray[(y - 1) * width + (x + 1)];

      sum += gray[y * width + (x - 1)] * 2;
      sum += gray[y * width + x] * 4;
      sum += gray[y * width + (x + 1)] * 2;

      sum += gray[(y + 1) * width + (x - 1)];
      sum += gray[(y + 1) * width + x] * 2;
      sum += gray[(y + 1) * width + (x + 1)];

      out[y * width + x] = Math.round(sum / 16);
    }
  }

  return out;
}

function detectFastLikeCorners(gray, width, height, options = {}) {
  const threshold = options.threshold ?? 30;
  const minArcLength = options.minArcLength ?? 9;

  const circleOffsets = [
    [0, -3], [1, -3], [2, -2], [3, -1],
    [3, 0], [3, 1], [2, 2], [1, 3],
    [0, 3], [-1, 3], [-2, 2], [-3, 1],
    [-3, 0], [-3, -1], [-2, -2], [-1, -3]
  ];

  const keypoints = [];

  for (let y = 4; y < height - 4; y++) {
    for (let x = 4; x < width - 4; x++) {
      const center = gray[y * width + x];
      const brighter = new Array(16);
      const darker = new Array(16);

      for (let k = 0; k < 16; k++) {
        const nx = x + circleOffsets[k][0];
        const ny = y + circleOffsets[k][1];
        const value = gray[ny * width + nx];

        brighter[k] = value >= center + threshold ? 1 : 0;
        darker[k] = value <= center - threshold ? 1 : 0;
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

  return keypoints;
}

function nonMaxSuppression(keypoints, options = {}) {
  const radius = options.radius ?? 10;
  const maxPoints = options.maxPoints ?? 150;
  const suppressed = [];

  const sorted = [...keypoints].sort((a, b) => b.score - a.score);

  for (const kp of sorted) {
    let keep = true;

    for (const chosen of suppressed) {
      const dx = kp.x - chosen.x;
      const dy = kp.y - chosen.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < radius) {
        keep = false;
        break;
      }
    }

    if (keep) {
      suppressed.push(kp);
    }

    if (suppressed.length >= maxPoints) {
      break;
    }
  }

  return suppressed;
}

function estimateOrientation(gray, width, height, x, y, patchRadius = 8) {
  let m10 = 0;
  let m01 = 0;

  for (let py = -patchRadius; py <= patchRadius; py++) {
    for (let px = -patchRadius; px <= patchRadius; px++) {
      const xx = x + px;
      const yy = y + py;

      if (xx < 0 || xx >= width || yy < 0 || yy >= height) {
        continue;
      }

      const intensity = gray[yy * width + xx];
      m10 += px * intensity;
      m01 += py * intensity;
    }
  }

  return Math.atan2(m01, m10);
}

function computeSimpleBinaryDescriptor(gray, width, height, x, y, angle) {
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const bits = [];

  for (const pair of DESCRIPTOR_PAIRS) {
    const ax = Math.round(x + pair.ax * cosA - pair.ay * sinA);
    const ay = Math.round(y + pair.ax * sinA + pair.ay * cosA);

    const bx = Math.round(x + pair.bx * cosA - pair.by * sinA);
    const by = Math.round(y + pair.bx * sinA + pair.by * cosA);

    if (
      ax < 0 || ax >= width || ay < 0 || ay >= height ||
      bx < 0 || bx >= width || by < 0 || by >= height
    ) {
      return null;
    }

    const a = gray[ay * width + ax];
    const b = gray[by * width + bx];
    bits.push(a < b ? 1 : 0);
  }

  return bits;
}

// Optional helper for future matching step
function hammingDistance(descA, descB) {
  if (!descA || !descB || descA.length !== descB.length) {
    return Infinity;
  }

  let distance = 0;
  for (let i = 0; i < descA.length; i++) {
    if (descA[i] !== descB[i]) {
      distance++;
    }
  }

  return distance;
}

function trackTargetInFrame(targetFeatures, frameFeatures) {
  const rawMatches = matchDescriptors(
    targetFeatures.keypoints,
    frameFeatures.keypoints,
    {
      maxDistance: MATCH_DISTANCE_THRESHOLD,
      ratioThreshold: MATCH_RATIO_THRESHOLD,
      crossCheck: true
    }
  );

  if (rawMatches.length < MIN_MATCH_COUNT) {
    return {
      found: false,
      rawMatches,
      inlierMatches: [],
      homography: null,
      projectedCorners: null
    };
  }

  const homographyResult = estimateHomographyRansac(
    rawMatches,
    RANSAC_ITERATIONS,
    RANSAC_INLIER_THRESHOLD
  );

  if (
    !homographyResult ||
    !homographyResult.homography ||
    homographyResult.inlierMatches.length < MIN_INLIER_COUNT
  ) {
    return {
      found: false,
      rawMatches,
      inlierMatches: homographyResult?.inlierMatches || [],
      homography: homographyResult?.homography || null,
      projectedCorners: null
    };
  }

  const projectedCorners = projectTargetCorners(
    targetFeatures.width,
    targetFeatures.height,
    homographyResult.homography
  );

  const validShape = isProjectedQuadReasonable(projectedCorners, frameFeatures.width, frameFeatures.height);

  return {
    found: validShape,
    rawMatches,
    inlierMatches: homographyResult.inlierMatches,
    homography: homographyResult.homography,
    projectedCorners: validShape ? projectedCorners : null
  };
}

function matchDescriptors(targetKeypoints, frameKeypoints, options = {}) {
  const maxDistance = options.maxDistance ?? 18;
  const ratioThreshold = options.ratioThreshold ?? 0.85;
  const crossCheck = options.crossCheck ?? true;

  const forwardMatches = [];

  for (let i = 0; i < targetKeypoints.length; i++) {
    const targetKp = targetKeypoints[i];

    let bestIndex = -1;
    let bestDistance = Infinity;
    let secondBestDistance = Infinity;

    for (let j = 0; j < frameKeypoints.length; j++) {
      const frameKp = frameKeypoints[j];
      const dist = hammingDistance(targetKp.descriptor, frameKp.descriptor);

      if (dist < bestDistance) {
        secondBestDistance = bestDistance;
        bestDistance = dist;
        bestIndex = j;
      } else if (dist < secondBestDistance) {
        secondBestDistance = dist;
      }
    }

    if (bestIndex === -1) {
      continue;
    }

    if (bestDistance > maxDistance) {
      continue;
    }

    if (secondBestDistance !== Infinity) {
      const ratio = bestDistance / Math.max(secondBestDistance, 1);
      if (ratio > ratioThreshold) {
        continue;
      }
    }

    forwardMatches.push({
      targetIndex: i,
      frameIndex: bestIndex,
      distance: bestDistance,
      targetPoint: {
        x: targetKp.x,
        y: targetKp.y
      },
      framePoint: {
        x: frameKeypoints[bestIndex].x,
        y: frameKeypoints[bestIndex].y
      }
    });
  }

  if (!crossCheck) {
    return forwardMatches;
  }

  const backwardBest = new Map();

  for (let j = 0; j < frameKeypoints.length; j++) {
    const frameKp = frameKeypoints[j];
    let bestIndex = -1;
    let bestDistance = Infinity;

    for (let i = 0; i < targetKeypoints.length; i++) {
      const targetKp = targetKeypoints[i];
      const dist = hammingDistance(frameKp.descriptor, targetKp.descriptor);

      if (dist < bestDistance) {
        bestDistance = dist;
        bestIndex = i;
      }
    }

    if (bestIndex !== -1) {
      backwardBest.set(j, bestIndex);
    }
  }

  const consistentMatches = forwardMatches.filter((match) => {
    return backwardBest.get(match.frameIndex) === match.targetIndex;
  });

  consistentMatches.sort((a, b) => a.distance - b.distance);
  return consistentMatches;
}

function estimateHomographyRansac(matches, iterations = 250, inlierThreshold = 18) {
  if (!matches || matches.length < 4) {
    return null;
  }

  let bestHomography = null;
  let bestInliers = [];

  for (let iter = 0; iter < iterations; iter++) {
    const sample = pickRandomUniqueMatches(matches, 4);
    if (sample.length < 4) {
      continue;
    }

    const H = computeHomographyDLT(sample);
    if (!H) {
      continue;
    }

    const inliers = [];

    for (const match of matches) {
      const projected = applyHomography(H, match.targetPoint.x, match.targetPoint.y);
      if (!projected) {
        continue;
      }

      const dx = projected.x - match.framePoint.x;
      const dy = projected.y - match.framePoint.y;
      const error = Math.sqrt(dx * dx + dy * dy);

      if (error <= inlierThreshold) {
        inliers.push(match);
      }
    }

    if (inliers.length > bestInliers.length) {
      bestInliers = inliers;
      bestHomography = H;
    }
  }

  if (!bestHomography || bestInliers.length < 4) {
    return null;
  }

  const refinedHomography = computeHomographyDLT(bestInliers) || bestHomography;

  return {
    homography: refinedHomography,
    inlierMatches: bestInliers
  };
}

function pickRandomUniqueMatches(matches, count) {
  if (matches.length < count) {
    return [];
  }

  const indices = new Set();
  while (indices.size < count) {
    const index = Math.floor(Math.random() * matches.length);
    indices.add(index);
  }

  return [...indices].map((i) => matches[i]);
}

function computeHomographyDLT(matches) {
  if (!matches || matches.length < 4) {
    return null;
  }

  const A = [];

  for (const match of matches) {
    const x = match.targetPoint.x;
    const y = match.targetPoint.y;
    const u = match.framePoint.x;
    const v = match.framePoint.y;

    A.push([-x, -y, -1, 0, 0, 0, x * u, y * u, u]);
    A.push([0, 0, 0, -x, -y, -1, x * v, y * v, v]);
  }

  const AtA = multiplyTransposeSelf(A);
  const eigenvector = smallestEigenvectorSymmetric(AtA, 60);

  if (!eigenvector) {
    return null;
  }

  const h = eigenvector;
  if (Math.abs(h[8]) < 1e-8) {
    return null;
  }

  return [
    [h[0] / h[8], h[1] / h[8], h[2] / h[8]],
    [h[3] / h[8], h[4] / h[8], h[5] / h[8]],
    [h[6] / h[8], h[7] / h[8], 1]
  ];
}

function multiplyTransposeSelf(A) {
  const rows = A.length;
  const cols = A[0].length;
  const out = Array.from({ length: cols }, () => Array(cols).fill(0));

  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < cols; j++) {
        out[i][j] += A[r][i] * A[r][j];
      }
    }
  }

  return out;
}

function multiplyMatrixVector(M, v) {
  const out = new Array(M.length).fill(0);

  for (let i = 0; i < M.length; i++) {
    let sum = 0;
    for (let j = 0; j < v.length; j++) {
      sum += M[i][j] * v[j];
    }
    out[i] = sum;
  }

  return out;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function normalizeVector(v) {
  const n = norm(v);
  if (n < 1e-12) {
    return null;
  }
  return v.map((x) => x / n);
}

function smallestEigenvectorSymmetric(M, iterations = 60) {
  const n = M.length;
  let b = new Array(n).fill(0).map(() => Math.random());
  b = normalizeVector(b);

  if (!b) {
    return null;
  }

  const shifted = M.map((row, i) =>
    row.map((value, j) => value + (i === j ? 1e-6 : 0))
  );

  for (let iter = 0; iter < iterations; iter++) {
    const x = solveLinearSystem(shifted, b);
    if (!x) {
      return null;
    }

    const normalized = normalizeVector(x);
    if (!normalized) {
      return null;
    }

    b = normalized;
  }

  return b;
}

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxAbs = Math.abs(M[col][col]);

    for (let r = col + 1; r < n; r++) {
      const value = Math.abs(M[r][col]);
      if (value > maxAbs) {
        maxAbs = value;
        pivotRow = r;
      }
    }

    if (maxAbs < 1e-12) {
      return null;
    }

    if (pivotRow !== col) {
      [M[col], M[pivotRow]] = [M[pivotRow], M[col]];
    }

    const pivot = M[col][col];
    for (let c = col; c <= n; c++) {
      M[col][c] /= pivot;
    }

    for (let r = 0; r < n; r++) {
      if (r === col) continue;

      const factor = M[r][col];
      for (let c = col; c <= n; c++) {
        M[r][c] -= factor * M[col][c];
      }
    }
  }

  return M.map((row) => row[n]);
}

function applyHomography(H, x, y) {
  const denom = H[2][0] * x + H[2][1] * y + H[2][2];

  if (Math.abs(denom) < 1e-8) {
    return null;
  }

  const u = (H[0][0] * x + H[0][1] * y + H[0][2]) / denom;
  const v = (H[1][0] * x + H[1][1] * y + H[1][2]) / denom;

  return { x: u, y: v };
}

function projectTargetCorners(targetWidth, targetHeight, H) {
  const corners = [
    { x: 0, y: 0 },
    { x: targetWidth, y: 0 },
    { x: targetWidth, y: targetHeight },
    { x: 0, y: targetHeight }
  ];

  return corners
    .map((p) => applyHomography(H, p.x, p.y))
    .filter(Boolean);
}

function isProjectedQuadReasonable(corners, frameWidth, frameHeight) {
  if (!corners || corners.length !== 4) {
    return false;
  }

  for (const p of corners) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) {
      return false;
    }
  }

  const area = polygonArea(corners);
  if (area < 400) {
    return false;
  }

  const xs = corners.map((p) => p.x);
  const ys = corners.map((p) => p.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  if (maxX < -50 || minX > frameWidth + 50 || maxY < -50 || minY > frameHeight + 50) {
    return false;
  }

  const width = maxX - minX;
  const height = maxY - minY;

  const edgeLengths = [
    Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y),
    Math.hypot(corners[2].x - corners[1].x, corners[2].y - corners[1].y),
    Math.hypot(corners[3].x - corners[2].x, corners[3].y - corners[2].y),
    Math.hypot(corners[0].x - corners[3].x, corners[0].y - corners[3].y)
  ];

  const maxEdge = Math.max(...edgeLengths);
  const minEdge = Math.min(...edgeLengths);

  if (minEdge < 10) return false;
  if (maxEdge / minEdge > 8) return false;

  if (width < 20 || height < 20) {
    return false;
  }

  return true;
}

function polygonArea(points) {
  let area = 0;

  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length;
    area += points[i].x * points[j].y - points[j].x * points[i].y;
  }

  return Math.abs(area) * 0.5;
}

function setupThreeScene() {
  if (!THREE_MODULE) {
    throw new Error("THREE_MODULE is not loaded");
  }

  if (!GLTFLoaderClass) {
    throw new Error("GLTFLoaderClass is not loaded");
  }

  if (!threeCanvas) {
    throw new Error("three-canvas element not found");
  }

  const THREE = THREE_MODULE;

  renderer = new THREE.WebGLRenderer({
    canvas: threeCanvas,
    alpha: true,
    antialias: true
  });

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(arView.clientWidth, arView.clientHeight);

  scene = new THREE.Scene();

  arCamera3D = new THREE.PerspectiveCamera(
    45,
    arView.clientWidth / arView.clientHeight,
    0.01,
    100
  );

  arCamera3D.position.z = 2;

  ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
  scene.add(ambientLight);

  directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
  directionalLight.position.set(1, 2, 3);
  scene.add(directionalLight);

  trackedObjectRoot = new THREE.Group();
  trackedObjectRoot.visible = false;
  scene.add(trackedObjectRoot);

  gltfLoader = new GLTFLoaderClass();
}

function resizeThreeScene() {
  if (!renderer || !arCamera3D) return;

  const width = arView.clientWidth;
  const height = arView.clientHeight;

  renderer.setSize(width, height);
  arCamera3D.aspect = width / height;
  arCamera3D.updateProjectionMatrix();
}

async function loadSelectedModel(modelKey) {
  if (!trackedObjectRoot) return;

  while (trackedObjectRoot.children.length > 0) {
    trackedObjectRoot.remove(trackedObjectRoot.children[0]);
  }

  activeModel = null;

  const modelPath = MODEL_PATHS[modelKey];
  if (!modelPath) return;

  try {
    const gltf = await gltfLoader.loadAsync(modelPath);
    activeModel = gltf.scene;

    const THREE = THREE_MODULE;
    const box = new THREE.Box3().setFromObject(activeModel);
    const center = box.getCenter(new THREE.Vector3());
    activeModel.position.sub(center);

    activeModel.scale.set(0.2, 0.2, 0.2);
    activeModel.rotation.x = -Math.PI / 2;

    trackedObjectRoot.add(activeModel);
  } catch (error) {
    console.error("Failed to load model:", error);
    throw error;
  }
}

function updateTracked3DObject(trackingResult) {
  if (!trackedObjectRoot || !activeModel || !arCamera3D || !latestFrameFeatures || !THREE_MODULE) {
    return;
  }

  if (!trackingResult || !trackingResult.found || !trackingResult.projectedCorners || trackingResult.projectedCorners.length !== 4) {
    trackedObjectRoot.visible = false;
    return;
  }

  const corners = trackingResult.projectedCorners;

  const center = {
    x: (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4,
    y: (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4
  };

  const ndcX = (center.x / latestFrameFeatures.width) * 2 - 1;
  const ndcY = -((center.y / latestFrameFeatures.height) * 2 - 1);

  const THREE = THREE_MODULE;

  // Put the model at a fixed depth in front of the camera
  const worldPos = new THREE.Vector3(ndcX, ndcY, 0.2).unproject(arCamera3D);

  trackedObjectRoot.visible = true;

  // Light smoothing only for position
  if (!smoothedPosition) {
    smoothedPosition = worldPos.clone();
  } else {
    smoothedPosition.lerp(worldPos, 0.25);
  }

  trackedObjectRoot.position.copy(smoothedPosition);

  // Fixed scale
  trackedObjectRoot.scale.setScalar(0.15);

  // No angle, no offset
  trackedObjectRoot.rotation.set(0, 0, 0);
}

function renderThreeScene() {
  if (!renderer || !scene || !arCamera3D) return;
  renderer.render(scene, arCamera3D);
}