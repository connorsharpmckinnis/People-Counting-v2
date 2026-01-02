const form = document.getElementById("uploadForm");
const resultsEl = document.getElementById("results");

const imgEl = document.getElementById("annotatedImage");
const videoEl = document.getElementById("annotatedVideo");
const typeSelect = document.getElementById("typeSelect");
const slicedImageSettings = document.getElementById("slicedImageSettings");
const slicedVideoSettings = document.getElementById("slicedVideoSettings");
const polygonCrossCountSettings = document.getElementById("polygonCrossCountSettings");
const resultsSection = document.getElementById("resultsSection");
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const filePreview = document.getElementById("filePreview");
const previewName = document.getElementById("previewName");
const removeFileBtn = document.getElementById("removeFileBtn");
const livestreamSettings = document.getElementById("livestreamSettings");
const imageGroup = document.getElementById("imageGroup");
const videoGroup = document.getElementById("videoGroup");

const confInput = document.getElementById("confInput");
const confValue = document.getElementById("confValue");
const btnText = document.getElementById("btnText");
const btnLoader = document.getElementById("btnLoader");
const downloadBtn = document.getElementById("downloadResultBtn");

const imageZoneSettings = document.getElementById("imageZoneSettings");
const imgZoneCanvas = document.getElementById("imgZoneCanvas");
const imgZonePlaceholder = document.getElementById("imgZonePlaceholder");
const imgZonePointsInput = document.getElementById("imgZonePoints");
const imgZoneClearBtn = document.getElementById("imgZoneClearBtn");
const imgZonePointCountSpan = document.getElementById("imgZonePointCount");

const modelSelect = document.getElementById("modelSelect");
const modelInput = document.getElementById("modelInput");
const standardClassesGroup = document.getElementById("standardClassesGroup");
const customClassesGroup = document.getElementById("customClassesGroup");
const customClassesInput = document.getElementById("customClassesInput");


// Update confidence value display in real-time
confInput.addEventListener("input", () => {
    confValue.textContent = confInput.value;
});

// Show/hide settings based on processing type
typeSelect.addEventListener("change", () => {
    // Hide all settings first
    slicedImageSettings.style.display = "none";
    slicedVideoSettings.style.display = "none";
    polygonCrossCountSettings.style.display = "none";
    imageZoneSettings.style.display = "none";

    // Show appropriate settings
    if (typeSelect.value === "sliced") {
        slicedImageSettings.style.display = "block";
    } else if (typeSelect.value === "sliced_video") {
        slicedVideoSettings.style.display = "block";
    } else if (typeSelect.value === "polygon_cross_count") {
        polygonCrossCountSettings.style.display = "block";
        // Trigger canvas update if file already selected
        if (fileInput.files.length > 0) {
            loadVideoFrame(fileInput.files[0]);
        }
    } else if (typeSelect.value === "image_zone_count") {
        imageZoneSettings.style.display = "block";
        if (fileInput.files.length > 0) {
            loadImageToCanvas(fileInput.files[0]);
        }
    } else if (typeSelect.value === "stream") {
        livestreamSettings.style.display = "block";
    }

    // Toggle drop zone and file requirement
    if (typeSelect.value === "stream") {
        dropZone.style.display = "none";
        fileInput.required = false;
        filePreview.style.display = "none";
    } else {
        if (!fileInput.files.length) {
            dropZone.style.display = "flex";
        }
        fileInput.required = true;
    }

    // Toggle between Standard (COCO IDs) and Custom (Strings) Class Inputs
    const isCustom = (typeSelect.value === "image_custom" || typeSelect.value === "video_custom");

    // 1. Toggle Input Groups
    standardClassesGroup.style.display = isCustom ? "none" : "block";
    customClassesGroup.style.display = isCustom ? "block" : "none";

    // 2. Manage Model Options (Mutually Exclusive)
    Array.from(modelSelect.options).forEach(opt => {
        if (opt.value === "yoloe-11s-seg.pt") {
            // YOLOE is ONLY for Custom
            opt.disabled = !isCustom;
        } else {
            // All other models are ONLY for Standard
            opt.disabled = isCustom;
        }
    });

    // 3. Auto-switch if selection is invalid
    // If we just switched to Custom, we force YOLOE.
    // If we switched to Standard, we force Nano (if current was YOLOE).
    if (isCustom) {
        if (modelSelect.value !== "yoloe-11s-seg.pt") {
            modelSelect.value = "yoloe-11s-seg.pt";
            modelInput.value = "yoloe-11s-seg.pt";
        }
    } else {
        if (modelSelect.value === "yoloe-11s-seg.pt") {
            modelSelect.value = "yolo11n.pt";
            modelInput.value = "yolo11n.pt";
        }
    }
});

// Initialize state on load
typeSelect.dispatchEvent(new Event('change'));

modelSelect.addEventListener("change", () => {
    modelInput.value = modelSelect.value;
});

// --- VIDEO REGION DRAWING LOGIC (Multi-Polygon Support) ---
const regionCanvas = document.getElementById("regionCanvas");
const canvasPlaceholder = document.getElementById("canvasPlaceholder");
const regionPointsInput = document.getElementById("regionPoints");
const clearPointsBtn = document.getElementById("clearPointsBtn");
const pointCountSpan = document.getElementById("pointCount");

// Multi-polygon data structure: array of arrays
let videoPolygons = [];       // Array of completed polygons
let currentVideoPoints = [];  // Points for the polygon currently being drawn
let videoResolution = { width: 0, height: 0 };
let currentFrameImage = null; // Store the frame for redrawing

// --- IMAGE ZONE DRAWING LOGIC (Multi-Polygon Support) ---
let imagePolygons = [];       // Array of completed polygons  
let currentImagePoints = [];  // Points for the polygon currently being drawn
let imgResolution = { width: 0, height: 0 };
let currentImgImage = null;

// Distinct colors for multiple polygons
const POLYGON_COLORS = [
    { stroke: "#00b894", fill: "rgba(0, 184, 148, 0.2)" },   // Green
    { stroke: "#0984e3", fill: "rgba(9, 132, 227, 0.2)" },   // Blue
    { stroke: "#e17055", fill: "rgba(225, 112, 85, 0.2)" },  // Orange
    { stroke: "#6c5ce7", fill: "rgba(108, 92, 231, 0.2)" },  // Purple
    { stroke: "#fdcb6e", fill: "rgba(253, 203, 110, 0.2)" }, // Yellow
    { stroke: "#e84393", fill: "rgba(232, 67, 147, 0.2)" },  // Pink
    { stroke: "#00cec9", fill: "rgba(0, 206, 201, 0.2)" },   // Cyan
    { stroke: "#ff7675", fill: "rgba(255, 118, 117, 0.2)" }, // Red
];

// --- FILE UPLOAD & FILTERING LOGIC ---

["dragover", "dragleave", "drop"].forEach(name => {
    dropZone.addEventListener(name, e => {
        e.preventDefault();
        if (name === "dragover") dropZone.classList.add("drop-zone--over");
        else dropZone.classList.remove("drop-zone--over");
    });
});

dropZone.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileChange(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length > 0) {
        handleFileChange(fileInput.files[0]);
    }
});

removeFileBtn.addEventListener("click", () => {
    fileInput.value = "";
    filePreview.style.display = "none";
    dropZone.style.display = "flex";
    imageGroup.disabled = false;
    videoGroup.disabled = false;
    resetVideoCanvas();
    resetImageCanvas();
    updateEstimation();
});

function handleFileChange(file) {
    if (!file) return;

    const isImage = file.type.startsWith("image/");
    const isVideo = file.type.startsWith("video/");

    // Update UI
    previewName.textContent = `📄 Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    filePreview.style.display = "block";
    dropZone.style.display = "none";

    // Filter Options
    imageGroup.disabled = !isImage;
    videoGroup.disabled = !isVideo;

    // Disable individual options for better visual feedback (Safari/some browsers ignore group disable)
    Array.from(imageGroup.options).forEach(opt => opt.disabled = !isImage);
    Array.from(videoGroup.options).forEach(opt => opt.disabled = !isVideo);

    // Auto-switch if current is invalid
    const currentIsImageMode = typeSelect.selectedOptions[0].parentElement.id === "imageGroup";
    const currentIsVideoMode = typeSelect.selectedOptions[0].parentElement.id === "videoGroup";

    if (isImage && !currentIsImageMode) {
        typeSelect.value = "basic";
    } else if (isVideo && !currentIsVideoMode) {
        typeSelect.value = "video";
    }

    typeSelect.dispatchEvent(new Event('change'));
    updateEstimation(); // Trigger estimation update

    // Loading Previews
    if (typeSelect.value === "polygon_cross_count" && isVideo) {
        loadVideoFrame(file);
    } else if (typeSelect.value === "image_zone_count" && isImage) {
        loadImageToCanvas(file);
    } else {
        resetVideoCanvas();
        resetImageCanvas();
    }
}

function loadVideoFrame(file) {
    const video = document.createElement("video");
    video.preload = "auto";
    video.src = URL.createObjectURL(file);
    video.muted = true;
    video.playsInline = true;

    // Use a flag to prevent multiple draws if seeked fires more than once
    let frameCaptured = false;

    video.onloadedmetadata = () => {
        videoResolution.width = video.videoWidth;
        videoResolution.height = video.videoHeight;

        // Set canvas internal resolution to match video
        regionCanvas.width = video.videoWidth;
        regionCanvas.height = video.videoHeight;

        // Seek a bit further in (0.5s) to avoid potentially empty initial frames
        video.currentTime = Math.min(0.5, video.duration || 0.5);
    };

    video.onseeked = () => {
        if (frameCaptured) return;

        // Use a small timeout to ensure the browser has actually rendered the frame
        // to the internal video buffer before we draw it to the canvas.
        setTimeout(() => {
            const ctx = regionCanvas.getContext("2d");
            ctx.drawImage(video, 0, 0, regionCanvas.width, regionCanvas.height);

            // Save frame for redrawing
            currentFrameImage = ctx.getImageData(0, 0, regionCanvas.width, regionCanvas.height);

            // Show canvas, hide placeholder
            regionCanvas.style.display = "block";
            canvasPlaceholder.style.display = "none";

            // Clear previous polygons for new video
            videoPolygons = [];
            currentVideoPoints = [];
            updateVideoRegionDisplay();

            frameCaptured = true;

            // Cleanup
            const blobUrl = video.src;
            video.src = "";
            video.load();
            setTimeout(() => URL.revokeObjectURL(blobUrl), 200);
        }, 300); // 300ms delay for decoder stability
    };

    video.onerror = () => {
        canvasPlaceholder.textContent = "Error loading video frame. The file format or codec may not be supported by your browser.";
        canvasPlaceholder.style.color = "#d63031";
    };
}

function loadImageToCanvas(file) {
    if (!file.type.startsWith("image/")) return;

    const img = new Image();
    img.onload = () => {
        imgResolution.width = img.width;
        imgResolution.height = img.height;
        imgZoneCanvas.width = img.width;
        imgZoneCanvas.height = img.height;

        const ctx = imgZoneCanvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        currentImgImage = ctx.getImageData(0, 0, imgZoneCanvas.width, imgZoneCanvas.height);

        imgZoneCanvas.style.display = "block";
        imgZonePlaceholder.style.display = "none";

        // Clear previous polygons for new image
        imagePolygons = [];
        currentImagePoints = [];
        updateImageRegionDisplay();

        URL.revokeObjectURL(img.src);
    };
    img.onerror = () => {
        imgZonePlaceholder.textContent = "Error loading image.";
    };
    img.src = URL.createObjectURL(file);
}

function resetVideoCanvas() {
    regionCanvas.style.display = "none";
    canvasPlaceholder.style.display = "block";
    videoPolygons = [];
    currentVideoPoints = [];
    updateVideoRegionDisplay();
}

function resetImageCanvas() {
    imgZoneCanvas.style.display = "none";
    imgZonePlaceholder.style.display = "block";
    imagePolygons = [];
    currentImagePoints = [];
    updateImageRegionDisplay();
}

// --- CLICK HANDLERS ---

// Single click: Add point to current polygon
regionCanvas.addEventListener("click", (e) => {
    const rect = regionCanvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0 || videoResolution.width === 0) return;

    const scaleX = videoResolution.width / rect.width;
    const scaleY = videoResolution.height / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    currentVideoPoints.push([x, y]);
    updateVideoRegionDisplay();
    redrawVideoCanvas();
});

imgZoneCanvas.addEventListener("click", (e) => {
    const rect = imgZoneCanvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0 || imgResolution.width === 0) return;

    const scaleX = imgResolution.width / rect.width;
    const scaleY = imgResolution.height / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    currentImagePoints.push([x, y]);
    updateImageRegionDisplay();
    redrawImageCanvas();
});

// --- FINISH REGION BUTTONS ---
// These finalize the current polygon and allow starting a new one

const finishVideoRegionBtn = document.getElementById("finishVideoRegionBtn");
const finishImageZoneBtn = document.getElementById("finishImageZoneBtn");

finishVideoRegionBtn.addEventListener("click", () => {
    if (currentVideoPoints.length >= 2) {
        videoPolygons.push([...currentVideoPoints]);
        currentVideoPoints = [];
        updateVideoRegionDisplay();
        redrawVideoCanvas();
    } else {
        alert("Please draw at least 2 points before finishing a region.");
    }
});

finishImageZoneBtn.addEventListener("click", () => {
    if (currentImagePoints.length >= 2) {
        imagePolygons.push([...currentImagePoints]);
        currentImagePoints = [];
        updateImageRegionDisplay();
        redrawImageCanvas();
    } else {
        alert("Please draw at least 2 points before finishing a zone.");
    }
});

// --- CLEAR BUTTONS ---
clearPointsBtn.addEventListener("click", () => {
    videoPolygons = [];
    currentVideoPoints = [];
    updateVideoRegionDisplay();
    redrawVideoCanvas();
});

imgZoneClearBtn.addEventListener("click", () => {
    imagePolygons = [];
    currentImagePoints = [];
    updateImageRegionDisplay();
    redrawImageCanvas();
});

// --- DISPLAY UPDATE FUNCTIONS ---

function updateVideoRegionDisplay() {
    // Count total points across all polygons + current
    const totalPoints = videoPolygons.reduce((sum, poly) => sum + poly.length, 0) + currentVideoPoints.length;
    const totalRegions = videoPolygons.length + (currentVideoPoints.length >= 2 ? 1 : 0);

    pointCountSpan.textContent = `${totalPoints} pts / ${videoPolygons.length} complete`;

    // Serialize as dictionary format for backend: {"region-01": [...], "region-02": [...]}
    const allPolygons = [...videoPolygons];
    if (currentVideoPoints.length >= 2) {
        allPolygons.push(currentVideoPoints);
    }

    if (allPolygons.length === 0) {
        regionPointsInput.value = "";
    } else if (allPolygons.length === 1) {
        // Single region - use list format for backward compatibility
        regionPointsInput.value = "[" + allPolygons[0].map(p => `(${p[0]}, ${p[1]})`).join(", ") + "]";
    } else {
        // Multiple regions - use dictionary format
        const regionDict = {};
        allPolygons.forEach((poly, idx) => {
            regionDict[`region-${String(idx + 1).padStart(2, '0')}`] = poly.map(p => `(${p[0]}, ${p[1]})`);
        });
        // Format as Python dict string
        let parts = [];
        for (const [key, pts] of Object.entries(regionDict)) {
            parts.push(`"${key}": [${pts.join(", ")}]`);
        }
        regionPointsInput.value = "{" + parts.join(", ") + "}";
    }
}

function updateImageRegionDisplay() {
    const totalPoints = imagePolygons.reduce((sum, poly) => sum + poly.length, 0) + currentImagePoints.length;

    imgZonePointCountSpan.textContent = `${totalPoints} pts / ${imagePolygons.length} complete`;

    const allPolygons = [...imagePolygons];
    if (currentImagePoints.length >= 2) {
        allPolygons.push(currentImagePoints);
    }

    if (allPolygons.length === 0) {
        imgZonePointsInput.value = "";
    } else if (allPolygons.length === 1) {
        imgZonePointsInput.value = "[" + allPolygons[0].map(p => `(${p[0]}, ${p[1]})`).join(", ") + "]";
    } else {
        const regionDict = {};
        allPolygons.forEach((poly, idx) => {
            regionDict[`region-${String(idx + 1).padStart(2, '0')}`] = poly.map(p => `(${p[0]}, ${p[1]})`);
        });
        let parts = [];
        for (const [key, pts] of Object.entries(regionDict)) {
            parts.push(`"${key}": [${pts.join(", ")}]`);
        }
        imgZonePointsInput.value = "{" + parts.join(", ") + "}";
    }
}

// --- DRAWING FUNCTIONS ---

function redrawVideoCanvas() {
    drawMultiPolygonCanvas(regionCanvas, currentFrameImage, videoPolygons, currentVideoPoints, videoResolution);
}

function redrawImageCanvas() {
    drawMultiPolygonCanvas(imgZoneCanvas, currentImgImage, imagePolygons, currentImagePoints, imgResolution);
}

function drawMultiPolygonCanvas(canvas, bgImage, completedPolygons, currentPoints, resolution) {
    if (!bgImage) return;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(bgImage, 0, 0);

    const scaleLine = Math.max(2, resolution.width / 400);
    const scaleRadius = Math.max(3, resolution.width / 200);

    // Draw all completed polygons
    completedPolygons.forEach((polygon, polyIndex) => {
        const colorIdx = polyIndex % POLYGON_COLORS.length;
        const color = POLYGON_COLORS[colorIdx];
        drawSinglePolygon(ctx, polygon, color, scaleLine, scaleRadius, true);
    });

    // Draw current in-progress polygon (with dashed lines to show it's incomplete)
    if (currentPoints.length > 0) {
        const colorIdx = completedPolygons.length % POLYGON_COLORS.length;
        const color = POLYGON_COLORS[colorIdx];
        drawSinglePolygon(ctx, currentPoints, color, scaleLine, scaleRadius, false);
    }
}

function drawSinglePolygon(ctx, points, color, scaleLine, scaleRadius, isComplete) {
    if (points.length === 0) return;

    // Draw lines/polygon
    if (points.length > 1) {
        ctx.beginPath();
        ctx.lineWidth = scaleLine;
        ctx.strokeStyle = color.stroke;

        if (!isComplete) {
            ctx.setLineDash([10, 5]); // Dashed line for in-progress
        } else {
            ctx.setLineDash([]); // Solid line for complete
        }

        ctx.moveTo(points[0][0], points[0][1]);

        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1]);
        }

        // Close polygon if > 2 points and complete
        if (points.length > 2 && isComplete) {
            ctx.lineTo(points[0][0], points[0][1]);
            ctx.fillStyle = color.fill;
            ctx.fill();
        }

        ctx.stroke();
        ctx.setLineDash([]); // Reset
    }

    // Draw points (vertices)
    ctx.fillStyle = color.stroke; // Use same color as stroke for points

    for (let i = 0; i < points.length; i++) {
        ctx.beginPath();
        ctx.arc(points[i][0], points[i][1], scaleRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    // Draw region label for complete polygons
    if (isComplete && points.length > 0) {
        // Calculate centroid for label placement
        const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
        const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;

        ctx.font = `bold ${Math.max(14, scaleLine * 8)}px Arial`;
        ctx.fillStyle = color.stroke;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        // We don't have the index here, so skip labeling for now
    }
}


form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const model = form.model.value;
    const conf = form.conf_threshold.value;
    const classInput = document.getElementById("polygonClasses").value.trim();

    // Show loading state
    btnText.style.display = "none";
    btnLoader.style.display = "inline-block";
    form.querySelector(".submit-btn").disabled = true;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("model", model);
    formData.append("conf_threshold", conf);
    // integer class parsing (Standard Modes)
    if (typeSelect.value !== "image_custom" && typeSelect.value !== "video_custom") {
        if (classInput) {
            const classes = classInput
                .split(",")
                .map(v => v.trim())
                .filter(v => v !== "")
                .map(v => Number(v))
                .filter(v => Number.isInteger(v) && v >= 0);

            if (classes.length > 0) {
                // Send as JSON string; FastAPI will parse it
                formData.append("classes", JSON.stringify(classes));
            }
        }
    } else if (typeSelect.value === "stream") {
        // Stream mode uses standard classes (integers)
        if (classInput) {
            const classes = classInput
                .split(",")
                .map(v => v.trim())
                .filter(v => v !== "")
                .map(v => Number(v))
                .filter(v => Number.isInteger(v) && v >= 0);

            if (classes.length > 0) {
                formData.append("classes", JSON.stringify(classes));
            }
        }
    } else {
        // user string parsing (Custom Modes)
        // pass the raw string from the custom input, backend handles list conversion
        const customClasses = customClassesInput.value.trim();
        if (customClasses) {
            formData.append("classes", customClasses);
        }
    }

    // Add sliced image settings if that mode is selected
    if (typeSelect.value === "sliced") {
        formData.append("slice_width", form.img_slice_width.value);
        formData.append("slice_height", form.img_slice_height.value);
        formData.append("overlap_width_ratio", form.img_overlap_width_ratio.value);
        formData.append("overlap_height_ratio", form.img_overlap_height_ratio.value);
    }

    // Add sliced video settings if that mode is selected
    if (typeSelect.value === "sliced_video") {
        formData.append("slice_width", form.slice_width.value);
        formData.append("slice_height", form.slice_height.value);
        formData.append("overlap_width", form.overlap_width.value);
        formData.append("overlap_height", form.overlap_height.value);
    }

    // Add polygon cross count settings if that mode is selected
    if (typeSelect.value === "polygon_cross_count") {
        formData.append("region_points", form.region_points.value);
    }

    // Add image zone count settings
    if (typeSelect.value === "image_zone_count") {
        formData.append("region_points", form.img_zone_points.value);
    }

    // Add stream settings
    if (typeSelect.value === "stream") {
        formData.append("youtube_url", form.youtube_url.value);
        formData.append("duration", form.duration.value);
        formData.append("frame_skip", form.frame_skip.value);
        // Clear file from formData if present (though we shouldn't have one selected)
        formData.delete("file");
    }

    let endpoint = "/basic-count";
    switch (typeSelect.value) {
        case "basic": endpoint = "/basic-count"; break;
        case "sliced": endpoint = "/sliced-count"; break;
        case "video": endpoint = "/video-count"; break;
        case "sliced_video": endpoint = "/sliced-video-count"; break;
        case "polygon_cross_count": endpoint = "/polygon-cross-count"; break;
        case "image_zone_count": endpoint = "/image-zone-count"; break;
        case "image_custom": endpoint = "/image-custom-count"; break;
        case "video_custom": endpoint = "/video-custom-count"; break;
        case "stream": endpoint = "/stream-count"; break;
    }

    resultsEl.textContent = "Processing your file...";
    imgEl.style.display = "none";
    videoEl.style.display = "none";
    downloadBtn.style.display = "none";
    resultsSection.style.display = "block";

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();

        // Handle job queue response
        if (data.job_id) {
            resultsEl.textContent = "Job submitted! Waiting in queue...";
            pollJobStatus(data.job_id);
        } else if (data.counts) {
            // Fallback for immediate response (though we shouldn't have any anymore)
            displayResults(data);
        }

    } catch (error) {
        resultsEl.textContent = `❌ Error: ${error.message}`;
        resultsEl.style.color = "#d63031";
        resetBtnState();
    }
});

function resetBtnState() {
    btnText.style.display = "inline";
    btnLoader.style.display = "none";
    form.querySelector(".submit-btn").disabled = false;
}

async function pollJobStatus(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}`);
        if (!response.ok) throw new Error("Failed to poll status");

        const job = await response.json();

        if (job.status === "completed") {
            displayResults(job.result);
            resetBtnState();
        } else if (job.status === "failed") {
            resultsEl.textContent = `❌ Job Failed: ${job.error}`;
            resultsEl.style.color = "#d63031";
            resetBtnState();
        } else {
            // Still running or queued
            if (job.status === "queued") {
                resultsEl.textContent = "⏳ Job is queued...";
                resultsEl.style.color = "#0984e3";
            } else if (job.status === "processing") {
                resultsEl.textContent = "⚙️ Processing... This may take a moment.";
                resultsEl.style.color = "#e17055";
            }

            // Poll again in 2 seconds
            setTimeout(() => pollJobStatus(jobId), 2000);
        }

    } catch (e) {
        console.error("Polling error:", e);
        resultsEl.textContent = "Error checking job status.";
        resetBtnState();
    }
}

function displayResults(data) {
    // Format the counts nicely
    if (Object.keys(data.counts).length === 0) {
        resultsEl.textContent = "No objects detected";
        resultsEl.style.color = "var(--text-primary)";
    } else {
        resultsEl.textContent = JSON.stringify(data.counts, null, 2);
        resultsEl.style.color = "var(--text-primary)";
    }

    // Show the appropriate media
    if (data.file_type === "image") {
        imgEl.src = data.annotated_file;
        imgEl.style.display = "block";
        videoEl.style.display = "none";
    } else if (data.file_type === "video") {
        videoEl.src = data.annotated_file;
        videoEl.load();
        videoEl.style.display = "block";
        imgEl.style.display = "none";
    }

    // Set download filename
    const originalFile = fileInput.files[0];
    const originalName = originalFile ? originalFile.name.split('.').slice(0, -1).join('.') : "result";
    const extension = data.file_type === "video" ? "mp4" : "png";
    downloadBtn.download = `${originalName}_processed.${extension}`;

    // Show download button
    downloadBtn.href = data.annotated_file;
    downloadBtn.style.display = "inline-block";

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Result copying logic
document.getElementById("copyResultsBtn").addEventListener("click", () => {
    const text = resultsEl.textContent;
    if (!text || text === "Processing..." || text.includes("No objects detected")) return;

    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById("copyResultsBtn");
        const originalText = btn.innerHTML;
        btn.innerHTML = "✅ Copied!";
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
});

// --- ESTIMATION LOGIC ---
const estimationSection = document.getElementById("estimationSection");
const estRes = document.getElementById("estRes");
const estFrames = document.getElementById("estFrames");
const estSlices = document.getElementById("estSlices");
const estTotal = document.getElementById("estTotal");
const estDuration = document.getElementById("estDuration");

const debouncedUpdate = debounce(updateEstimation, 500);

// Add listeners
fileInput.addEventListener("change", updateEstimation);
typeSelect.addEventListener("change", updateEstimation);

// Sliced Image Inputs
["imgSliceWidth", "imgSliceHeight", "imgOverlapWidthRatio", "imgOverlapHeightRatio"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", debouncedUpdate);
});

// Sliced Video Inputs
["sliceWidth", "sliceHeight", "overlapWidth", "overlapHeight"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", debouncedUpdate);
});


async function updateEstimation() {
    const file = fileInput.files[0];
    if (!file || typeSelect.value === "stream") {
        estimationSection.style.display = "none";
        return;
    }

    estimationSection.style.display = "block";
    estRes.textContent = "Calculating...";
    estFrames.textContent = "...";
    estSlices.textContent = "...";
    estTotal.textContent = "...";
    estDuration.textContent = "";

    const formData = new FormData();
    formData.append("file", file);

    // Add relevant settings based on type
    // Note: We send all slice params if available, backend logic filters what it needs
    // or we can be specific. Let's send what's relevant to current view.

    if (typeSelect.value === "sliced") {
        formData.append("slice_width", document.getElementById("imgSliceWidth").value);
        formData.append("slice_height", document.getElementById("imgSliceHeight").value);
        formData.append("overlap_width_ratio", document.getElementById("imgOverlapWidthRatio").value);
        formData.append("overlap_height_ratio", document.getElementById("imgOverlapHeightRatio").value);
    } else if (typeSelect.value === "sliced_video") {
        formData.append("slice_width", document.getElementById("sliceWidth").value);
        formData.append("slice_height", document.getElementById("sliceHeight").value);
        formData.append("overlap_width", document.getElementById("overlapWidth").value);
        formData.append("overlap_height", document.getElementById("overlapHeight").value);
    }

    try {
        const response = await fetch("/estimate-count", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Estimation failed");

        const data = await response.json();

        if (data.error) {
            estRes.textContent = "Error";
            estDuration.textContent = data.error;
            return;
        }

        estRes.textContent = `${data.resolution[0]} x ${data.resolution[1]}`;
        estFrames.textContent = data.total_frames;
        estSlices.textContent = data.slices_per_frame;
        estTotal.textContent = data.total_inference_steps.toLocaleString();

        if (data.duration_seconds) {
            estDuration.textContent = `Duration: ${data.duration_seconds}s @ ${data.fps}fps`;
        } else {
            estDuration.textContent = "";
        }

        // --- WARNING LOGIC ---
        const totalSteps = data.total_inference_steps;
        const estWarning = document.getElementById("estWarning");

        if (totalSteps > 1000) {
            estWarning.style.display = "block";
            estWarning.style.color = "#d63031"; // red
            estWarning.innerHTML = "⚠️ Whoa there! over 1,000 inferences? Are you trying to melt the server? (This will take a long time)";
        } else if (totalSteps > 100) {
            estWarning.style.display = "block";
            estWarning.style.color = "#fdcb6e"; // orange/yellow
            estWarning.style.color = "#e17055"; // darker orange for readability
            estWarning.textContent = "⏳ This might take a minute or two...";
        } else {
            estWarning.style.display = "none";
        }

    } catch (e) {
        console.error("Estimation error:", e);
        estRes.textContent = "-";
        estDuration.textContent = "Could not estimate (server error)";
    }
}

function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}
