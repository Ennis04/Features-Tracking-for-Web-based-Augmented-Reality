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

  // Placeholder for future generate function
  isGenerated = true;
  enableSaveButton();
  statusMessage.textContent = "Features generated successfully. You can now save the result.";
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
  previewPlaceholder.style.display = "block";
  fileNameText.textContent = "No file selected";
  actionButtons.classList.add("hidden");
  statusMessage.textContent = "";
  uploadedImageData = "";
  isGenerated = false;
  disableSaveButton();
}