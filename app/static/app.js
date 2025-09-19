const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileLabel = document.getElementById("file-label");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("result");
const transcriptEl = document.getElementById("transcript");

const STATUS_VARIANTS = {
  info: "info",
  loading: "loading",
  success: "success",
  error: "error",
};

function setStatus(message, variant = STATUS_VARIANTS.info) {
  statusEl.textContent = message;
  statusEl.dataset.variant = variant;
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length === 0) {
    fileLabel.textContent = "Choose an audio file";
    setStatus("Select a file to begin.");
    resultSection.hidden = true;
    return;
  }

  const [file] = fileInput.files;
  fileLabel.textContent = file.name;
  setStatus(`Ready to upload ${file.name}`);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (fileInput.files.length === 0) {
    setStatus("Please choose an audio or video file first.", STATUS_VARIANTS.error);
    return;
  }

  const [file] = fileInput.files;
  const formData = new FormData();
  formData.append("file", file);

  setStatus("Uploading and transcribing…", STATUS_VARIANTS.loading);
  resultSection.hidden = true;

  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      const detail = errorBody?.detail ?? "Unable to transcribe the supplied file.";
      throw new Error(detail);
    }

    const { text } = await response.json();
    transcriptEl.textContent = text?.trim() || "No transcript returned.";
    resultSection.hidden = false;
    setStatus("Transcription complete!", STATUS_VARIANTS.success);
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Something went wrong while transcribing.", STATUS_VARIANTS.error);
  }
});
