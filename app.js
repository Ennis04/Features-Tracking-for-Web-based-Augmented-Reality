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
const arStatusMessage = document.getElementById("ar-status-message");
const arOverlay = document.getElementById("ar-overlay");
const threeCanvas = document.getElementById("three-canvas");

const SHOW_ALL_FRAME_KEYPOINTS = true;
const FRAME_PROCESS_INTERVAL = 120;
const FRAME_MAX_WIDTH = 640;

const MATCH_MAX_DISTANCE = 75;
const MIN_MATCH_COUNT = 8;
const MIN_INLIER_COUNT = 6;
const MATCH_DISTANCE_THRESHOLD = 22;
const MATCH_RATIO_THRESHOLD = 0.9;

const MODEL_PATHS = {
  sphere: "model/sphere/scene.gltf",
  cube: "model/cube/scene.gltf",
  pyramid: "model/pyramid/scene.gltf",
  cuboid: "model/cuboid/scene.gltf"
};

const frameCanvas = document.createElement("canvas");
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });

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
let trackedObjectRoot = null;
let activeModel = null;
let gltfLoader = null;

let smoothedPosition = null;

let lastGoodTrackingResult = null;
let lostFrameCount = 0;
const MAX_LOST_FRAMES = 8;

/* ---------------------------
   FRIEND-STYLE TRACKER CORE
---------------------------- */

class CustomTracker {
  static briefPairs = null;

  static getGrayscale(imageData) {
    const { width, height, data } = imageData;
    const gray = new Uint8Array(width * height);

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = Math.round(
        0.299 * data[i] +
        0.587 * data[i + 1] +
        0.114 * data[i + 2]
      );
    }

    return { width, height, data: gray };
  }

  static detectFAST(grayImg, threshold = 35) {
    const { width, height, data } = grayImg;
    const circle = [
      [0, -3], [1, -3], [2, -2], [3, -1],
      [3, 0], [3, 1], [2, 2], [1, 3],
      [0, 3], [-1, 3], [-2, 2], [-3, 1],
      [-3, 0], [-3, -1], [-2, -2], [-1, -3]
    ];

    const gridSize = 12;
    const grid = new Map();

    for (let y = 4; y < height - 4; y += 2) {
      for (let x = 4; x < width - 4; x += 2) {
        const center = data[y * width + x];

        // quick reject using 4 points
        let quick = 0;
        const quickPts = [0, 4, 8, 12];
        for (const idx of quickPts) {
          const nx = x + circle[idx][0];
          const ny = y + circle[idx][1];
          const v = data[ny * width + nx];
          if (Math.abs(v - center) > threshold) quick++;
        }
        if (quick < 3) continue;

        const brighter = new Array(16).fill(0);
        const darker = new Array(16).fill(0);

        for (let k = 0; k < 16; k++) {
          const nx = x + circle[k][0];
          const ny = y + circle[k][1];
          const v = data[ny * width + nx];
          brighter[k] = v >= center + threshold ? 1 : 0;
          darker[k] = v <= center - threshold ? 1 : 0;
        }

        const br2 = brighter.concat(brighter);
        const dk2 = darker.concat(darker);

        let isCorner = false;
        for (let start = 0; start < 16; start++) {
          let bCount = 0;
          let dCount = 0;
          for (let j = 0; j < 9; j++) {
            bCount += br2[start + j];
            dCount += dk2[start + j];
          }
          if (bCount === 9 || dCount === 9) {
            isCorner = true;
            break;
          }
        }

        if (!isCorner) continue;

        let score = 0;
        for (let k = 0; k < 16; k++) {
          const nx = x + circle[k][0];
          const ny = y + circle[k][1];
          const v = data[ny * width + nx];
          score += Math.abs(v - center);
        }

        const cellX = Math.floor(x / gridSize);
        const cellY = Math.floor(y / gridSize);
        const key = `${cellX},${cellY}`;
        const existing = grid.get(key);

        if (!existing || score > existing.score) {
          grid.set(key, { x, y, score });
        }
      }
    }

    return [...grid.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, 180);
  }

  static seededRandom(seed) {
    let s = seed >>> 0;
    return function () {
      s = (1664525 * s + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  static ensureBriefPairs() {
    if (this.briefPairs) return;

    const rand = this.seededRandom(123456789);
    this.briefPairs = [];

    for (let i = 0; i < 256; i++) {
      const ax = Math.round(rand() * 30 - 15);
      const ay = Math.round(rand() * 30 - 15);
      const bx = Math.round(rand() * 30 - 15);
      const by = Math.round(rand() * 30 - 15);
      this.briefPairs.push({ ax, ay, bx, by });
    }
  }

  static popcnt8(v) {
    v = v - ((v >> 1) & 0x55);
    v = (v & 0x33) + ((v >> 2) & 0x33);
    return (((v + (v >> 4)) & 0x0f) * 0x01);
  }

  static hammingPacked(a, b) {
    let d = 0;
    for (let i = 0; i < a.length; i++) {
      d += this.popcnt8(a[i] ^ b[i]);
    }
    return d;
  }

  static computeBRIEF(grayImg, keypoints) {
    this.ensureBriefPairs();

    const { width, height, data } = grayImg;
    const descriptors = [];
    const validKeypoints = [];

    for (const kp of keypoints) {
      const x = kp.x;
      const y = kp.y;

      if (x < 16 || y < 16 || x >= width - 16 || y >= height - 16) {
        continue;
      }

      const bytes = new Uint8Array(32);

      for (let i = 0; i < 256; i++) {
        const p = this.briefPairs[i];

        const ax = x + p.ax;
        const ay = y + p.ay;
        const bx = x + p.bx;
        const by = y + p.by;

        const av = data[ay * width + ax];
        const bv = data[by * width + bx];

        if (av < bv) {
          bytes[i >> 3] |= (1 << (i & 7));
        }
      }

      validKeypoints.push(kp);
      descriptors.push(bytes);
    }

    return {
      keypoints: validKeypoints,
      descriptors
    };
  }

  static matchFeatures(descA, descB, maxDistance = 75, ratio = 0.7) {
    const matches = [];

    for (let i = 0; i < descA.length; i++) {
      let bestJ = -1;
      let bestDist = Infinity;
      let secondDist = Infinity;

      for (let j = 0; j < descB.length; j++) {
        const d = this.hammingPacked(descA[i], descB[j]);

        if (d < bestDist) {
          secondDist = bestDist;
          bestDist = d;
          bestJ = j;
        } else if (d < secondDist) {
          secondDist = d;
        }
      }

      if (bestJ === -1) continue;
      if (bestDist > maxDistance) continue;
      if (secondDist !== Infinity && bestDist / Math.max(secondDist, 1) > ratio) continue;

      matches.push({
        idxA: i,
        idxB: bestJ,
        distance: bestDist
      });
    }

    matches.sort((a, b) => a.distance - b.distance);
    return matches;
  }

  static solveAffineFromTwoPairs(p1, p2, q1, q2, cx, cy) {
    const p1x = p1.x - cx;
    const p1y = p1.y - cy;
    const p2x = p2.x - cx;
    const p2y = p2.y - cy;

    const dpX = p2x - p1x;
    const dpY = p2y - p1y;
    const dqX = q2.x - q1.x;
    const dqY = q2.y - q1.y;

    const lenP = Math.hypot(dpX, dpY);
    const lenQ = Math.hypot(dqX, dqY);

    if (lenP < 1e-6 || lenQ < 1e-6) return null;

    const scale = lenQ / lenP;
    const angleP = Math.atan2(dpY, dpX);
    const angleQ = Math.atan2(dqY, dqX);
    const angle = angleQ - angleP;

    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    const tx = q1.x - scale * (p1x * cosA - p1y * sinA);
    const ty = q1.y - scale * (p1x * sinA + p1y * cosA);

    return { scale, angle, tx, ty };
  }

  static solveAffineRansac(markerKp, frameKp, matches, cx, cy) {
    if (!matches || matches.length < 2) return null;

    let bestModel = null;
    let bestInliers = [];

    const iterations = Math.min(250, matches.length * 20);

    for (let iter = 0; iter < iterations; iter++) {
      const i1 = Math.floor(Math.random() * matches.length);
      let i2 = Math.floor(Math.random() * matches.length);
      while (i2 === i1) i2 = Math.floor(Math.random() * matches.length);

      const m1 = matches[i1];
      const m2 = matches[i2];

      const p1 = markerKp[m1.idxA];
      const p2 = markerKp[m2.idxA];
      const q1 = frameKp[m1.idxB];
      const q2 = frameKp[m2.idxB];

      const model = this.solveAffineFromTwoPairs(p1, p2, q1, q2, cx, cy);
      if (!model) continue;

      const cosA = Math.cos(model.angle);
      const sinA = Math.sin(model.angle);
      const inliers = [];

      for (const m of matches) {
        const p = markerKp[m.idxA];
        const q = frameKp[m.idxB];

        const lx = p.x - cx;
        const ly = p.y - cy;

        const px = model.scale * (lx * cosA - ly * sinA) + model.tx;
        const py = model.scale * (lx * sinA + ly * cosA) + model.ty;

        const err = Math.hypot(px - q.x, py - q.y);
        if (err < 10) {
          inliers.push(m);
        }
      }

      if (inliers.length > bestInliers.length) {
        bestInliers = inliers;
        bestModel = model;
      }
    }

    if (!bestModel || bestInliers.length < MIN_INLIER_COUNT) {
      return null;
    }

    // refine from inliers
    let sumScale = 0;
    let sumAngle = 0;
    let sumTx = 0;
    let sumTy = 0;
    let count = 0;

    for (let i = 0; i < bestInliers.length - 1; i++) {
      const m1 = bestInliers[i];
      const m2 = bestInliers[i + 1];

      const model = this.solveAffineFromTwoPairs(
        markerKp[m1.idxA],
        markerKp[m2.idxA],
        frameKp[m1.idxB],
        frameKp[m2.idxB],
        cx,
        cy
      );

      if (!model) continue;

      sumScale += model.scale;
      sumAngle += model.angle;
      sumTx += model.tx;
      sumTy += model.ty;
      count++;
    }

    if (count > 0) {
      bestModel = {
        scale: sumScale / count,
        angle: sumAngle / count,
        tx: sumTx / count,
        ty: sumTy / count
      };
    }

    return {
      ...bestModel,
      inliers: bestInliers
    };
  }
}

/* ---------------------------
   THREE LOADING
---------------------------- */

async function ensureThreeLoaded() {
  if (THREE_MODULE && GLTFLoaderClass) return;

  const threeImport = await import("https://esm.sh/three@0.160.0");
  const loaderImport = await import("https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js");

  THREE_MODULE = threeImport;
  GLTFLoaderClass = loaderImport.GLTFLoader;
}

/* ---------------------------
   UI EVENTS
---------------------------- */

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
      "No stable feature points were found. Try another image with more texture or contrast.";
    return;
  }

  drawFeaturePreview(targetFeatures);

  isGenerated = true;
  enableActionButtons();

  statusMessage.textContent =
    `Feature analysis completed successfully. ${targetFeatures.keypoints.length} keypoints extracted.`;
});

linkModelBtn.addEventListener("click", function () {
  if (!isGenerated || !targetFeatures) {
    statusMessage.textContent = "Please click Generate before selecting a 3D model.";
    return;
  }

  modelPanel.classList.remove("hidden");
  statusMessage.textContent = "Please choose one of the predefined 3D model options.";
});

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
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false
      });
    } catch {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
    }

    arVideo.srcObject = webcamStream;
    arView.classList.remove("hidden");
    document.body.style.overflow = "hidden";

    await new Promise((resolve, reject) => {
      arVideo.onloadedmetadata = () => resolve();
      arVideo.onerror = () => reject(new Error("Video metadata failed to load"));
    });

    await arVideo.play();
    setupArOverlay();

    try {
      await ensureThreeLoaded();
      setupThreeScene();
      resizeThreeScene();
    } catch (threeError) {
      console.error("Three.js setup failed:", threeError);
      statusMessage.textContent = `Camera started, but 3D setup failed: ${threeError.message}`;
      return;
    }

    try {
      if (selectedModel) {
        await loadSelectedModel(selectedModel);
      }
    } catch (modelError) {
      console.error("Model load failed:", modelError);
      statusMessage.textContent = `Camera started, but model loading failed: ${modelError.message}`;
      return;
    }

    startArProcessingLoop();
    statusMessage.textContent = `AR view started with model: ${selectedModel}.`;
  } catch (error) {
    console.error("Start AR failed:", error);
    statusMessage.textContent = `Start AR failed: ${error.name || "Error"} - ${error.message || ""}`;
  }
});

backBtn.addEventListener("click", function () {
  closeArView();
});

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

/* ---------------------------
   UI HELPERS
---------------------------- */

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
  scene = null;
  arCamera3D = null;
  activeModel = null;
  gltfLoader = null;
}

function setStatus(message) {
  if (statusMessage) {
    statusMessage.textContent = message;
  }

  if (arStatusMessage) {
    arStatusMessage.textContent = message;
  }
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

/* ---------------------------
   FEATURE EXTRACTION
---------------------------- */

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
  const grayImg = CustomTracker.getGrayscale(imageData);
  const keypoints = CustomTracker.detectFAST(grayImg, 25);
  const brief = CustomTracker.computeBRIEF(grayImg, keypoints);

  return {
    width,
    height,
    gray: grayImg,
    keypoints: brief.keypoints,
    descriptors: brief.descriptors,
    cx: width / 2,
    cy: height / 2
  };
}

function extractFrameFeaturesFromVideo(video) {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (!sourceWidth || !sourceHeight) return null;

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
  const grayImg = CustomTracker.getGrayscale(imageData);
  const keypoints = CustomTracker.detectFAST(grayImg, 25);
  const brief = CustomTracker.computeBRIEF(grayImg, keypoints);

  return {
    width,
    height,
    gray: grayImg,
    keypoints: brief.keypoints,
    descriptors: brief.descriptors
  };
}

function drawFeaturePreview(featureData) {
  const { width, height, gray, keypoints } = featureData;
  const ctx = previewCanvas.getContext("2d");

  previewCanvas.width = width;
  previewCanvas.height = height;

  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < gray.data.length; i++) {
    const v = gray.data[i];
    const base = i * 4;
    imageData.data[base] = v;
    imageData.data[base + 1] = v;
    imageData.data[base + 2] = v;
    imageData.data[base + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);

  ctx.fillStyle = "red";
  for (const kp of keypoints) {
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  showCanvasPreview();
}

/* ---------------------------
   TRACKING
---------------------------- */

function trackTargetInFrame(targetFeatures, frameFeatures) {
  const rawMatches = CustomTracker.matchFeatures(
    targetFeatures.descriptors,
    frameFeatures.descriptors,
    MATCH_MAX_DISTANCE
  );

  if (rawMatches.length < MIN_MATCH_COUNT) {
    return {
      found: false,
      rawMatches: [],
      inlierMatches: [],
      affineModel: null,
      projectedCorners: null
    };
  }

  const model = CustomTracker.solveAffineRansac(
    targetFeatures.keypoints,
    frameFeatures.keypoints,
    rawMatches,
    targetFeatures.cx,
    targetFeatures.cy
  );

  if (!model || !model.inliers || model.inliers.length < MIN_INLIER_COUNT) {
    return {
      found: false,
      rawMatches: mapMatchesForOverlay(rawMatches, targetFeatures.keypoints, frameFeatures.keypoints),
      inlierMatches: [],
      affineModel: null,
      projectedCorners: null
    };
  }

  const projectedCorners = projectCornersFromAffine(
    targetFeatures.width,
    targetFeatures.height,
    model,
    targetFeatures.cx,
    targetFeatures.cy
  );

  return {
    found: true,
    rawMatches: mapMatchesForOverlay(rawMatches, targetFeatures.keypoints, frameFeatures.keypoints),
    inlierMatches: mapMatchesForOverlay(model.inliers, targetFeatures.keypoints, frameFeatures.keypoints),
    affineModel: model,
    projectedCorners
  };
}

function mapMatchesForOverlay(matches, targetKeypoints, frameKeypoints) {
  return matches.map((m) => ({
    targetPoint: {
      x: targetKeypoints[m.idxA].x,
      y: targetKeypoints[m.idxA].y
    },
    framePoint: {
      x: frameKeypoints[m.idxB].x,
      y: frameKeypoints[m.idxB].y
    },
    distance: m.distance
  }));
}

function projectCornersFromAffine(width, height, model, cx, cy) {
  const corners = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height }
  ];

  const cosA = Math.cos(model.angle);
  const sinA = Math.sin(model.angle);

  return corners.map((p) => {
    const lx = p.x - cx;
    const ly = p.y - cy;

    return {
      x: model.scale * (lx * cosA - ly * sinA) + model.tx,
      y: model.scale * (lx * sinA + ly * cosA) + model.ty
    };
  });
}

/* ---------------------------
   LOOP + OVERLAY
---------------------------- */

function startArProcessingLoop() {
  if (isArRunning) return;

  isArRunning = true;
  lastFrameProcessTime = 0;
  latestTrackingResult = null;

  if (latestTrackingResult && latestTrackingResult.found) {
    lastGoodTrackingResult = latestTrackingResult;
    lostFrameCount = 0;
  } else {
    lostFrameCount++;
    if (lostFrameCount <= MAX_LOST_FRAMES && lastGoodTrackingResult) {
      latestTrackingResult = lastGoodTrackingResult;
    }
  }

  const loop = (timestamp) => {
    if (!isArRunning) return;

    arLoopId = requestAnimationFrame(loop);

    if (!arVideo.videoWidth || !arVideo.videoHeight) {
      return;
    }

    if (timestamp - lastFrameProcessTime < FRAME_PROCESS_INTERVAL) {
      drawArOverlay();
      renderThreeScene();
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
    updateTrackingStatus(latestFrameFeatures, latestTrackingResult);
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

function drawArOverlay() {
  const ctx = arOverlay.getContext("2d");
  ctx.clearRect(0, 0, arOverlay.width, arOverlay.height);

  if (!latestFrameFeatures) return;

  const scaleX = arOverlay.width / latestFrameFeatures.width;
  const scaleY = arOverlay.height / latestFrameFeatures.height;

  if (SHOW_ALL_FRAME_KEYPOINTS) {
    ctx.fillStyle = "rgba(255, 80, 80, 0.9)";
    for (const kp of latestFrameFeatures.keypoints) {
      const x = kp.x * scaleX;
      const y = kp.y * scaleY;
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (!latestTrackingResult) return;

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
  }
}

function updateTrackingStatus(frameFeatures, trackingResult) {
  if (!frameFeatures) return;

  const targetCount = targetFeatures?.keypoints.length || 0;
  const frameCount = frameFeatures.keypoints.length;

  if (!trackingResult) {
    setStatus(
      `AR live loop running
Target keypoints: ${targetCount}
Frame keypoints: ${frameCount}
Matches: 0`
    );
    return;
  }

  const rawMatches = trackingResult.rawMatches.length;
  const inliers = trackingResult.inlierMatches.length;

  if (trackingResult.found) {
    setStatus(
      `Target FOUND
Target keypoints: ${targetCount}
Frame keypoints: ${frameCount}
Raw matches: ${rawMatches}
Inliers: ${inliers}`
    );
  } else {
    setStatus(
      `Tracking...
Target keypoints: ${targetCount}
Frame keypoints: ${frameCount}
Raw matches: ${rawMatches}
Inliers: ${inliers}`
    );
  }
}

/* ---------------------------
   THREE / MODEL
---------------------------- */

function setupThreeScene() {
  if (!THREE_MODULE) throw new Error("THREE_MODULE is not loaded");
  if (!GLTFLoaderClass) throw new Error("GLTFLoaderClass is not loaded");
  if (!threeCanvas) throw new Error("three-canvas element not found");

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

  const gltf = await gltfLoader.loadAsync(modelPath);
  activeModel = gltf.scene;

  const THREE = THREE_MODULE;
  const box = new THREE.Box3().setFromObject(activeModel);
  const center = box.getCenter(new THREE.Vector3());
  activeModel.position.sub(center);

  // make it MUCH bigger for debugging
  activeModel.scale.set(0.4, 0.4, 0.4);
  activeModel.rotation.set(0, 0, 0);

  trackedObjectRoot.add(activeModel);
}

function updateTracked3DObject(trackingResult) {
  if (!trackedObjectRoot || !activeModel || !arCamera3D || !latestFrameFeatures || !THREE_MODULE) {
    return;
  }

  const THREE = THREE_MODULE;

  // If tracking is good, place model at tracked center
  if (trackingResult && trackingResult.found && trackingResult.projectedCorners && trackingResult.projectedCorners.length === 4) {
    const corners = trackingResult.projectedCorners;

    const center = {
      x: (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4,
      y: (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4
    };

    const ndcX = (center.x / latestFrameFeatures.width) * 2 - 1;
    const ndcY = -((center.y / latestFrameFeatures.height) * 2 - 1) + 1;

    // place a bit in front of camera
    const worldPos = new THREE.Vector3(ndcX, ndcY, 0).unproject(arCamera3D);

    trackedObjectRoot.visible = true;
    trackedObjectRoot.position.copy(worldPos);

    // bigger scale for debugging
    trackedObjectRoot.scale.setScalar(0.4);
    trackedObjectRoot.rotation.set(0, 0, 0);
    return;
  }

  // fallback: still show model using raw matches if available
  if (trackingResult && trackingResult.rawMatches && trackingResult.rawMatches.length >= 3) {
    let sumX = 0;
    let sumY = 0;

    for (const match of trackingResult.rawMatches) {
      sumX += match.framePoint.x;
      sumY += match.framePoint.y;
    }

    const centerX = sumX / trackingResult.rawMatches.length;
    const centerY = sumY / trackingResult.rawMatches.length;

    const ndcX = (centerX / latestFrameFeatures.width) * 2 - 1;
    const ndcY = -((centerY / latestFrameFeatures.height) * 2 - 1) + 1;

    const worldPos = new THREE.Vector3(ndcX, ndcY, 0).unproject(arCamera3D);

    trackedObjectRoot.visible = true;
    trackedObjectRoot.position.copy(worldPos);
    trackedObjectRoot.scale.setScalar(0.4);
    trackedObjectRoot.rotation.set(0, 0, 0);
    return;
  }

  trackedObjectRoot.visible = false;
}

function renderThreeScene() {
  if (!renderer || !scene || !arCamera3D) return;
  renderer.render(scene, arCamera3D);
}