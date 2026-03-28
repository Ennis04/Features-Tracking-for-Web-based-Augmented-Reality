const imageUpload = document.getElementById("image-upload");
const imagePreview = document.getElementById("image-preview");
const grayscaleCanvas = document.getElementById("grayscale-canvas");
const featureCanvas = document.getElementById("feature-canvas");

const originalPlaceholder = document.getElementById("original-placeholder");
const grayscalePlaceholder = document.getElementById("grayscale-placeholder");
const featurePlaceholder = document.getElementById("feature-placeholder");

const fileNameText = document.getElementById("file-name");
const actionButtons = document.getElementById("action-buttons");
const generateBtn = document.getElementById("generate-btn");
const startArBtn = document.getElementById("start-ar-btn");
const saveBtn = document.getElementById("save-btn");
const statusMessage = document.getElementById("status-message");

const modelSelect = document.getElementById("model-select");

const arView = document.getElementById("ar-view");
const arVideo = document.getElementById("ar-video");
const backBtn = document.getElementById("back-btn");
const arStatusMessage = document.getElementById("ar-status-message");
const arOverlay = document.getElementById("ar-overlay");
const threeCanvas = document.getElementById("three-canvas");

const SHOW_TRACKING_OVERLAY = false;
const FRAME_PROCESS_INTERVAL = 60;
const FRAME_MAX_WIDTH = 640;

const MATCH_MAX_DISTANCE = 75;
const MIN_MATCH_COUNT = 8;
const MIN_INLIER_COUNT = 6;

const MAX_LOST_FRAMES = 4;

const frameCanvas = document.createElement("canvas");
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });

let THREE_MODULE = null;
let GLTFLoaderClass = null;

let uploadedImageData = "";
let originalImage = new Image();
let webcamStream = null;
let targetFeatures = null;
let isGenerated = false;

let currentModelKey = "jett_knife";

let arLoopId = null;
let isArRunning = false;
let lastFrameProcessTime = 0;
let latestFrameFeatures = null;
let latestTrackingResult = null;
let lastGoodTrackingResult = null;
let lostFrameCount = 0;

let renderer = null;
let scene = null;
let arCamera3D = null;
let trackedObjectRoot = null;
let activeModel = null;
let gltfLoader = null;

// Configuration for size and location of the 3D models
const MODEL_CONFIGS = {
  jett_knife: {
    label: "Jett Knife",
    path: "model/jett_knife/scene.gltf",

    modelScale: [0.4, 0.4, 0.4],
    modelRotation: [0, 0, Math.PI / 2],
    modelPosition: [0, 0, 0],

    rootScale: 0.075,
    rootRotation: [0, 0, 0],
    rootPositionOffset: [0, 0, 0],

    autoSpinY: 0.04
  },

  cuboid: {
    label: "Cuboid",
    path: "model/cuboid/scene.gltf",

    modelScale: [1.5, 1.5, 1.5],
    modelRotation: [0, 0, 0],
    modelPosition: [0, 0, 0],

    rootScale: 0.07,
    rootRotation: [0, 0, 0],
    rootPositionOffset: [0, 0, 0],

    autoSpinY: 0.01
  },

  pyramid: {
    label: "Pyramid",
    path: "model/pyramid/scene.gltf",

    modelScale: [0.025, 0.05, 0.025],
    modelRotation: [0, 0, 0],
    modelPosition: [0, 0, 0],

    rootScale: 0.07,
    rootRotation: [0, 0, 0],
    rootPositionOffset: [0, 0, 0],

    autoSpinY: 0.01
  }
};

function getCurrentModelConfig() {
  return MODEL_CONFIGS[currentModelKey] || MODEL_CONFIGS.jett_knife;
}

// Feature detection using FAST + BRIEF
class FeatureTracker {
  static briefPairs = null;

  // Convert image into grayscale
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

  // Detect corners using FAST algorithm
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

  // Generate random pairs of points for BRIEF descriptor
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
    return ((v + (v >> 4)) & 0x0f) * 0x01;
  }

  // Compute Hamming distance between two BRIEF descriptors
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
          bytes[i >> 3] |= 1 << (i & 7);
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

  // Match features between two sets of descriptors
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

  // Estimate affine transform from two pairs of matched points
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

  // Robustly estimate affine transform using RANSAC
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

// Load Three.js and GLTFLoader when AR is started
async function ThreeLoaded() {
  if (THREE_MODULE && GLTFLoaderClass) return;

  const threeImport = await import("https://esm.sh/three@0.160.0");
  const loaderImport = await import("https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js");

  THREE_MODULE = threeImport;
  GLTFLoaderClass = loaderImport.GLTFLoader;
}

// Upload Image
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
      clearGeneratedPreviews();
      actionButtons.classList.remove("hidden");

      isGenerated = false;
      targetFeatures = null;
      disableGeneratedButtons();

      setStatus(
        "Image uploaded successfully. Click Generate to extract target features."
      );
    };

    originalImage.src = uploadedImageData;
  };

  reader.readAsDataURL(file);
});

// Select 3D model
modelSelect.addEventListener("change", async function () {
  currentModelKey = this.value;
  const cfg = getCurrentModelConfig();

  setStatus(`Selected 3D model: ${cfg.label}`);

  if (isArRunning && trackedObjectRoot && gltfLoader) {
      await loadModel();
      setStatus(`Model switched to: ${cfg.label}`);
  }
});

// Generate feature
generateBtn.addEventListener("click", function () {
  if (!uploadedImageData) {
    setStatus("Please upload an image first.");
    return;
  }

  targetFeatures = extractTargetFeatures(originalImage);

  if (!targetFeatures || targetFeatures.keypoints.length === 0) {
    isGenerated = false;
    targetFeatures = null;
    disableGeneratedButtons();
    clearGeneratedPreviews();
    setStatus("No feature points found.");
    return;
  }

  drawFeaturePreview(targetFeatures);

  isGenerated = true;
  enableGeneratedButtons();

  setStatus(
    `${targetFeatures.keypoints.length} keypoints extracted.`
  );
});

// Start AR
startArBtn.addEventListener("click", async function () {
  if (startArBtn.disabled) return;

  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" } },
    audio: false
  });

  arVideo.srcObject = webcamStream;
  arView.classList.remove("hidden");
  document.body.style.overflow = "hidden";

  await new Promise((resolve, reject) => {
    arVideo.onloadedmetadata = () => resolve();
    arVideo.onerror = () => reject(new Error("Video metadata failed to load"));
  });

  await arVideo.play();
  setupArOverlay();

  await ThreeLoaded();
  setupThreeScene();
  resizeThreeScene();
  await loadModel();

  startArProcessingLoop();
});

// Back button to navigate back to home page
backBtn.addEventListener("click", function () {
  closeArView();
});

// Save generated feature preview as image
saveBtn.addEventListener("click", function () {
  const link = document.createElement("a");
  link.href = featureCanvas.toDataURL("image/png");
  link.download = "feature-analysis-result.png";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  setStatus("Generated image saved to your local device.");
});

function showOriginalPreview() {
  originalPlaceholder.style.display = "none";
  imagePreview.style.display = "block";
  imagePreview.src = uploadedImageData;
}

function showGrayscalePreview() {
  grayscalePlaceholder.style.display = "none";
  grayscaleCanvas.style.display = "block";
}

function showFeaturePreview() {
  featurePlaceholder.style.display = "none";
  featureCanvas.style.display = "block";
}

function clearGeneratedPreviews() {
  grayscaleCanvas.style.display = "none";
  featureCanvas.style.display = "none";

  grayscalePlaceholder.style.display = "block";
  featurePlaceholder.style.display = "block";

  const gctx = grayscaleCanvas.getContext("2d");
  const fctx = featureCanvas.getContext("2d");

  gctx.clearRect(0, 0, grayscaleCanvas.width || 1, grayscaleCanvas.height || 1);
  fctx.clearRect(0, 0, featureCanvas.width || 1, featureCanvas.height || 1);
}

function disableGeneratedButtons() {
  saveBtn.disabled = true;
  saveBtn.classList.add("btn-disabled");

  startArBtn.disabled = true;
  startArBtn.classList.add("btn-disabled");
}

function enableGeneratedButtons() {
  saveBtn.disabled = false;
  saveBtn.classList.remove("btn-disabled");

  startArBtn.disabled = false;
  startArBtn.classList.remove("btn-disabled");
}

function setStatus(message) {
  statusMessage.textContent = message;
  arStatusMessage.textContent = message;
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

  scene = null;
  arCamera3D = null;
  trackedObjectRoot = null;
  activeModel = null;
  gltfLoader = null;
  latestFrameFeatures = null;
  latestTrackingResult = null;
  lastGoodTrackingResult = null;
  lostFrameCount = 0;
}

function resetPreview() {
  closeArView();

  imagePreview.src = "";
  imagePreview.style.display = "none";

  originalPlaceholder.style.display = "block";
  clearGeneratedPreviews();

  fileNameText.textContent = "No file selected";
  actionButtons.classList.add("hidden");
  statusMessage.textContent = "";
  uploadedImageData = "";
  isGenerated = false;
  targetFeatures = null;

  modelSelect.value = "jett_knife";
  currentModelKey = "jett_knife";

  disableGeneratedButtons();
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

// Extract features from the uploaded target image
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
  const grayImg = FeatureTracker.getGrayscale(imageData);
  const keypoints = FeatureTracker.detectFAST(grayImg, 25);
  const brief = FeatureTracker.computeBRIEF(grayImg, keypoints);

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

// Extract features from a video frame
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
  const grayImg = FeatureTracker.getGrayscale(imageData);
  const keypoints = FeatureTracker.detectFAST(grayImg, 25);
  const brief = FeatureTracker.computeBRIEF(grayImg, keypoints);

  return {
    width,
    height,
    gray: grayImg,
    keypoints: brief.keypoints,
    descriptors: brief.descriptors
  };
}

// Draw feature detection results on the preview
function drawFeaturePreview(featureData) {
  const { width, height, gray, keypoints } = featureData;
  drawGrayscalePreview(width, height, gray);
  drawFeaturePointsPreview(width, height, gray, keypoints);
}

// Draw grayscale image and detected keypoints on the preview
function drawGrayscalePreview(width, height, gray) {
  const ctx = grayscaleCanvas.getContext("2d");

  grayscaleCanvas.width = width;
  grayscaleCanvas.height = height;

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
  showGrayscalePreview();
}

// Draw feature points on top of the grayscale image in the preview
function drawFeaturePointsPreview(width, height, gray, keypoints) {
  const ctx = featureCanvas.getContext("2d");

  featureCanvas.width = width;
  featureCanvas.height = height;

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

  showFeaturePreview();
}

// Match target features with frame features and estimate the target's position in the video frame
function trackTargetInFrame(targetFeaturesData, frameFeatures) {
  const rawMatches = FeatureTracker.matchFeatures(
    targetFeaturesData.descriptors,
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

  const model = FeatureTracker.solveAffineRansac(
    targetFeaturesData.keypoints,
    frameFeatures.keypoints,
    rawMatches,
    targetFeaturesData.cx,
    targetFeaturesData.cy
  );

  if (!model || !model.inliers || model.inliers.length < MIN_INLIER_COUNT) {
    return {
      found: false,
      rawMatches: mapMatchesForOverlay(rawMatches, targetFeaturesData.keypoints, frameFeatures.keypoints),
      inlierMatches: [],
      affineModel: null,
      projectedCorners: null
    };
  }

  const projectedCorners = projectCornersFromAffine(
    targetFeaturesData.width,
    targetFeaturesData.height,
    model,
    targetFeaturesData.cx,
    targetFeaturesData.cy
  );

  return {
    found: true,
    rawMatches: mapMatchesForOverlay(rawMatches, targetFeaturesData.keypoints, frameFeatures.keypoints),
    inlierMatches: mapMatchesForOverlay(model.inliers, targetFeaturesData.keypoints, frameFeatures.keypoints),
    affineModel: model,
    projectedCorners
  };
}

// Convert matched feature indices into actual point coordinates for overlay visualization
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

// Use the estimated affine transform to project the corners of the target image onto the video frame for AR overlay
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

// Capture video frames, extract features, perform tracking, and update the AR overlay and 3D model position in a loop
function startArProcessingLoop() {
  if (isArRunning) return;

  isArRunning = true;
  lastFrameProcessTime = 0;
  latestTrackingResult = null;
  lastGoodTrackingResult = null;
  lostFrameCount = 0;

  const loop = (timestamp) => {
    if (!isArRunning) return;

    arLoopId = requestAnimationFrame(loop);

    if (!arVideo.videoWidth || !arVideo.videoHeight) {
      return;
    }

    if (timestamp - lastFrameProcessTime >= FRAME_PROCESS_INTERVAL) {
      lastFrameProcessTime = timestamp;
      latestFrameFeatures = extractFrameFeaturesFromVideo(arVideo);

      if (targetFeatures && latestFrameFeatures) {
        const currentResult = trackTargetInFrame(targetFeatures, latestFrameFeatures);

        if (currentResult.found) {
          latestTrackingResult = currentResult;
          lastGoodTrackingResult = currentResult;
          lostFrameCount = 0;
        } else if (lastGoodTrackingResult && lostFrameCount < MAX_LOST_FRAMES) {
          lostFrameCount++;
          latestTrackingResult = lastGoodTrackingResult;
        } else {
          latestTrackingResult = currentResult;
          lostFrameCount = MAX_LOST_FRAMES;
        }
      } else {
        latestTrackingResult = null;
      }

      updateTrackingStatus(latestFrameFeatures, latestTrackingResult);
      updateTracked3DObject(latestTrackingResult);
    }

    drawArOverlay();
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

// Draw tracking results and feature points on the AR overlay canvas for visualization and debugging
function drawArOverlay() {
  const ctx = arOverlay.getContext("2d");
  ctx.clearRect(0, 0, arOverlay.width, arOverlay.height);

  if (!SHOW_TRACKING_OVERLAY) return;
  if (!latestFrameFeatures) return;

  const scaleX = arOverlay.width / latestFrameFeatures.width;
  const scaleY = arOverlay.height / latestFrameFeatures.height;

  ctx.fillStyle = "rgba(255, 80, 80, 0.9)";
  for (const kp of latestFrameFeatures.keypoints) {
    ctx.beginPath();
    ctx.arc(kp.x * scaleX, kp.y * scaleY, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  if (!latestTrackingResult) return;

  if (latestTrackingResult.inlierMatches.length > 0) {
    ctx.strokeStyle = "rgba(0, 255, 255, 0.9)";
    ctx.lineWidth = 2;

    for (const match of latestTrackingResult.inlierMatches) {
      ctx.beginPath();
      ctx.arc(match.framePoint.x * scaleX, match.framePoint.y * scaleY, 4, 0, Math.PI * 2);
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

// Update the status message with current tracking information, including the number of keypoints, matches, and whether the target is currently found or being tracked
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

// Initialize Three.js scene, camera, lights, and GLTF loader for rendering the 3D model in AR
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

  const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
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

function clearTrackedObjectRoot() {
  if (!trackedObjectRoot) return;

  while (trackedObjectRoot.children.length > 0) {
    const child = trackedObjectRoot.children[0];
    trackedObjectRoot.remove(child);

    if (child.traverse) {
      child.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose?.();

        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((mat) => mat.dispose?.());
          } else {
            obj.material.dispose?.();
          }
        }
      });
    }
  }
}

// Load the selected 3D model
async function loadModel() {
  if (!trackedObjectRoot || !gltfLoader || !THREE_MODULE) return;

  const THREE = THREE_MODULE;
  const cfg = getCurrentModelConfig();

  clearTrackedObjectRoot();
  activeModel = null;

  const gltf = await gltfLoader.loadAsync(cfg.path);
  activeModel = gltf.scene;

  const box = new THREE.Box3().setFromObject(activeModel);
  const center = box.getCenter(new THREE.Vector3());

  activeModel.position.sub(center);

  activeModel.scale.set(
    cfg.modelScale[0],
    cfg.modelScale[1],
    cfg.modelScale[2]
  );

  activeModel.rotation.set(
    cfg.modelRotation[0],
    cfg.modelRotation[1],
    cfg.modelRotation[2]
  );

  activeModel.position.x += cfg.modelPosition[0];
  activeModel.position.y += cfg.modelPosition[1];
  activeModel.position.z += cfg.modelPosition[2];

  trackedObjectRoot.scale.setScalar(cfg.rootScale);
  trackedObjectRoot.rotation.set(
    cfg.rootRotation[0],
    cfg.rootRotation[1],
    cfg.rootRotation[2]
  );

  trackedObjectRoot.add(activeModel);
}

function updateTracked3DObject(trackingResult) {
  if (!trackedObjectRoot || !activeModel || !arCamera3D || !latestFrameFeatures || !THREE_MODULE) {
    return;
  }

  const THREE = THREE_MODULE;
  const cfg = getCurrentModelConfig();

  if (trackingResult && trackingResult.found && trackingResult.projectedCorners?.length === 4) {
    const corners = trackingResult.projectedCorners;

    const center = {
      x: (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4,
      y: (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4
    };

    const ndcX = (center.x / latestFrameFeatures.width) * 2 - 0.8;
    const ndcY = -(center.y / latestFrameFeatures.height) * 2 + 1;

    const worldPos = new THREE.Vector3(ndcX, ndcY, 0).unproject(arCamera3D);

    trackedObjectRoot.visible = true;
    trackedObjectRoot.position.copy(worldPos);

    trackedObjectRoot.position.x += cfg.rootPositionOffset[0];
    trackedObjectRoot.position.y += cfg.rootPositionOffset[1];
    trackedObjectRoot.position.z += cfg.rootPositionOffset[2];

    trackedObjectRoot.scale.setScalar(cfg.rootScale);
    trackedObjectRoot.rotation.set(
      cfg.rootRotation[0],
      cfg.rootRotation[1],
      cfg.rootRotation[2]
    );
    return;
  }

  if (trackingResult && trackingResult.rawMatches && trackingResult.rawMatches.length >= 3) {
    let sumX = 0;
    let sumY = 0;

    for (const match of trackingResult.rawMatches) {
      sumX += match.framePoint.x;
      sumY += match.framePoint.y;
    }

    const centerX = sumX / trackingResult.rawMatches.length;
    const centerY = sumY / trackingResult.rawMatches.length;

    const ndcX = (centerX / latestFrameFeatures.width) * 2 - 0.8;
    const ndcY = -(centerY / latestFrameFeatures.height) * 2 + 1;

    const worldPos = new THREE.Vector3(ndcX, ndcY, 0).unproject(arCamera3D);

    trackedObjectRoot.visible = true;
    trackedObjectRoot.position.copy(worldPos);

    trackedObjectRoot.position.x += cfg.rootPositionOffset[0];
    trackedObjectRoot.position.y += cfg.rootPositionOffset[1];
    trackedObjectRoot.position.z += cfg.rootPositionOffset[2];

    trackedObjectRoot.scale.setScalar(cfg.rootScale);
    trackedObjectRoot.rotation.set(
      cfg.rootRotation[0],
      cfg.rootRotation[1],
      cfg.rootRotation[2]
    );
    return;
  }

  trackedObjectRoot.visible = false;
}

function renderThreeScene() {
  if (!renderer || !scene || !arCamera3D) return;

  const cfg = getCurrentModelConfig();

  if (activeModel && cfg.autoSpinY) {
    activeModel.rotation.y += cfg.autoSpinY;
  }

  renderer.render(scene, arCamera3D);
}