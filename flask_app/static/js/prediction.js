/**
 * Prediction & Grad-CAM Workbench JavaScript
 * Handles client-side drag-and-drop, sample loading, AJAX inference, and visualization.
 */

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("uploadDropzone");
    const fileInput = document.getElementById("leafFileInput");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const modelSelect = document.getElementById("modelSelect");
    const gradcamToggle = document.getElementById("gradcamToggle");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultsContainer = document.getElementById("resultsContainer");
    const sampleCards = document.querySelectorAll(".sample-card");

    let currentSelectedFile = null;
    let currentSampleFilename = null;

    // 1. Setup Dropzone Click & Drag/Drop
    if (dropzone && fileInput) {
        dropzone.addEventListener("click", () => fileInput.click());

        ["dragenter", "dragover"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleSelectedFile(files[0]);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleSelectedFile(e.target.files[0]);
            }
        });
    }

    // 2. Handle File Selection
    function handleSelectedFile(file) {
        if (!file.type.startsWith("image/")) {
            showToast("Please select a valid image file (PNG, JPG, JPEG, WEBP).", "danger");
            return;
        }

        currentSelectedFile = file;
        currentSampleFilename = null;

        // Clear sample active states
        sampleCards.forEach(card => card.classList.remove("active"));

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = "block";
            analyzeBtn.disabled = false;
            // Scroll smoothly to preview
            previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };
        reader.readAsDataURL(file);
    }

    // 3. Handle Sample Leaf Selection
    sampleCards.forEach(card => {
        card.addEventListener("click", () => {
            sampleCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");

            currentSampleFilename = card.dataset.filename;
            currentSelectedFile = null;

            // Update preview image from thumbnail or fetch
            const img = card.querySelector("img");
            imagePreview.src = img.src;
            previewContainer.style.display = "block";
            analyzeBtn.disabled = false;

            // Trigger prediction automatically on sample selection for snappy UX
            runPrediction();
        });
    });

    // 4. Analyze Button Click
    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", () => {
            runPrediction();
        });
    }

    // 5. Run Prediction Pipeline
    async function runPrediction() {
        if (!currentSelectedFile && !currentSampleFilename) {
            showToast("Please upload an image or select a sample leaf first.", "danger");
            return;
        }

        const selectedModel = modelSelect.value;
        const enableGradcam = gradcamToggle ? gradcamToggle.checked : true;

        // Set Loading State
        analyzeBtn.disabled = true;
        const originalBtnHtml = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML = `<div class="spinner" style="width: 18px; height: 18px; border-width: 2px; display: inline-block; vertical-align: middle; margin-right: 8px;"></div> Analyzing Leaf...`;

        try {
            let response, data;

            if (currentSelectedFile) {
                // Upload Form Data
                const formData = new FormData();
                formData.append("file", currentSelectedFile);
                formData.append("model", selectedModel);
                formData.append("gradcam", enableGradcam ? "true" : "false");

                response = await fetch("/api/predict", {
                    method: "POST",
                    body: formData
                });
            } else if (currentSampleFilename) {
                // Sample Data
                response = await fetch("/api/predict-sample", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: currentSampleFilename,
                        model: selectedModel,
                        gradcam: enableGradcam
                    })
                });
            }

            data = await response.json();

            if (!data.success) {
                throw new Error(data.error || "Inference failed.");
            }

            renderPredictionResults(data.data);
            resultsContainer.style.display = "block";
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error("Prediction error:", error);
            showToast(error.message, "danger");
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = originalBtnHtml;
        }
    }

    // 6. Render Results into UI
    function renderPredictionResults(res) {
        // Predicted Header & Badge
        const predTitle = document.getElementById("resPredictedTitle");
        const predBadge = document.getElementById("resPredictedBadge");
        const confidencePct = document.getElementById("resConfidencePct");
        const latencyMs = document.getElementById("resLatencyMs");
        const modelUsed = document.getElementById("resModelUsed");
        const deviceUsed = document.getElementById("resDeviceUsed");

        if (predTitle) predTitle.textContent = res.display_name;
        if (predBadge) {
            predBadge.textContent = res.disease_info.pathogen_type || "Pathogen";
            predBadge.className = res.is_healthy ? "pill pill-green" : "pill pill-red";
        }
        if (confidencePct) confidencePct.textContent = res.confidence_pct;
        if (latencyMs) latencyMs.textContent = `${res.inference_time_ms} ms`;
        if (modelUsed) modelUsed.textContent = res.model_name;
        if (deviceUsed) deviceUsed.textContent = res.device;

        // Top-3 Confidence Bars
        const topKContainer = document.getElementById("resTopKContainer");
        if (topKContainer && res.top_predictions) {
            topKContainer.innerHTML = res.top_predictions.map((item, index) => {
                const fillClass = index === 0 ? "fill-top1" : (index === 1 ? "fill-top2" : "fill-top3");
                return `
                    <div class="confidence-bar-container">
                        <div class="confidence-header">
                            <span><strong>${index + 1}.</strong> ${item.display_name} <small class="text-muted">(${item.pathogen_type})</small></span>
                            <span>${item.confidence_pct}</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill ${fillClass}" style="width: ${item.percentage_num}%"></div>
                        </div>
                    </div>
                `;
            }).join("");
        }

        // Disease Diagnostic Information
        const diagScientific = document.getElementById("resScientificName");
        const diagSymptoms = document.getElementById("resSymptoms");
        const diagManagement = document.getElementById("resManagement");

        if (diagScientific) diagScientific.textContent = res.disease_info.scientific_name || "N/A";
        if (diagSymptoms) diagSymptoms.textContent = res.disease_info.symptoms || "N/A";
        if (diagManagement) diagManagement.textContent = res.disease_info.management || "N/A";

        // Grad-CAM Visualizations
        const camOverlay = document.getElementById("camOverlayImg");
        const camHeatmap = document.getElementById("camHeatmapImg");
        const camOriginal = document.getElementById("camOriginalImg");
        const camSection = document.getElementById("camSection");

        if (res.gradcam_superimposed_b64) {
            if (camOverlay) camOverlay.src = res.gradcam_superimposed_b64;
            if (camHeatmap) camHeatmap.src = res.gradcam_heatmap_b64;
            if (camOriginal) camOriginal.src = res.original_image_b64;
            if (camSection) camSection.style.display = "block";
            setupGradcamTabs();
        } else {
            if (camSection) camSection.style.display = "none";
        }
    }

    // 7. Setup Grad-CAM Tab Switching
    function setupGradcamTabs() {
        const tabBtns = document.querySelectorAll(".cam-tab-btn");
        const tabContents = {
            "overlay": document.getElementById("camOverlayContainer"),
            "heatmap": document.getElementById("camHeatmapContainer"),
            "original": document.getElementById("camOriginalContainer")
        };

        tabBtns.forEach(btn => {
            btn.onclick = () => {
                tabBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const target = btn.dataset.tab;

                Object.keys(tabContents).forEach(key => {
                    if (tabContents[key]) {
                        tabContents[key].style.display = (key === target) ? "block" : "none";
                    }
                });
            };
        });
    }
});
