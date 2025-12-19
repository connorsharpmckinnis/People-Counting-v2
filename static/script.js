const form = document.getElementById("uploadForm");
const resultsEl = document.getElementById("results");

const imgEl = document.getElementById("annotatedImage");
const videoEl = document.getElementById("annotatedVideo");
const typeSelect = document.getElementById("typeSelect");
const slicedImageSettings = document.getElementById("slicedImageSettings");
const slicedVideoSettings = document.getElementById("slicedVideoSettings");
const polygonCrossCountSettings = document.getElementById("polygonCrossCountSettings");
const resultsSection = document.getElementById("resultsSection");
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

// --- VIDEO REGION DRAWING LOGIC ---
const regionCanvas = document.getElementById("regionCanvas");
const canvasPlaceholder = document.getElementById("canvasPlaceholder");
const regionPointsInput = document.getElementById("regionPoints");
const clearPointsBtn = document.getElementById("clearPointsBtn");
const pointCountSpan = document.getElementById("pointCount");
let currentPoints = [];
let videoResolution = { width: 0, height: 0 };
let currentFrameImage = null; // Store the frame for redrawing

// --- IMAGE ZONE DRAWING LOGIC ---
let imgZonePoints = [];
let imgResolution = { width: 0, height: 0 };
let currentImgImage = null;

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) {
        resetCanvas();
        resetImgZoneCanvas();
        return;
    }

    if (typeSelect.value === "polygon_cross_count" && file.type.startsWith("video/")) {
        loadVideoFrame(file);
    } else if (typeSelect.value === "image_zone_count" && file.type.startsWith("image/")) {
        loadImageToCanvas(file);
    } else {
        resetCanvas();
        resetImgZoneCanvas();
    }
});

function loadVideoFrame(file) {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.src = URL.createObjectURL(file);
    video.muted = true;
    video.playsInline = true;

    video.onloadedmetadata = () => {
        videoResolution.width = video.videoWidth;
        videoResolution.height = video.videoHeight;

        // Set canvas internal resolution to match video
        regionCanvas.width = video.videoWidth;
        regionCanvas.height = video.videoHeight;

        // Seek to 0.1s to ensure we have a frame (sometimes 0.0 is empty)
        video.currentTime = 0.1;
    };

    video.onseeked = () => {
        // Draw frame to canvas
        const ctx = regionCanvas.getContext("2d");
        ctx.drawImage(video, 0, 0, regionCanvas.width, regionCanvas.height);

        // Save frame for redrawing
        currentFrameImage = ctx.getImageData(0, 0, regionCanvas.width, regionCanvas.height);

        // Show canvas, hide placeholder
        regionCanvas.style.display = "block";
        canvasPlaceholder.style.display = "none";

        // Clear previous points for new video
        currentPoints = [];
        updatePoints();

        // Cleanup: vital to stop the video element from trying to buffer more data
        // which causes the ERR_FILE_NOT_FOUND error if we just revoke immediately
        const blobUrl = video.src;
        video.src = "";
        video.load(); // stops the stream
        setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
    };

    video.onerror = () => {
        canvasPlaceholder.textContent = "Error loading video frame. Please check format.";
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

        imgZonePoints = [];
        updateImgZonePoints();

        URL.revokeObjectURL(img.src);
    };
    img.onerror = () => {
        imgZonePlaceholder.textContent = "Error loading image.";
    };
    img.src = URL.createObjectURL(file);
}

function resetCanvas() {
    regionCanvas.style.display = "none";
    canvasPlaceholder.style.display = "block";
    currentPoints = [];
    updatePoints();
}

function resetImgZoneCanvas() {
    imgZoneCanvas.style.display = "none";
    imgZonePlaceholder.style.display = "block";
    imgZonePoints = [];
    updateImgZonePoints();
}

// Handler for Video Canvas
regionCanvas.addEventListener("click", (e) => {
    handleCanvasClick(e, regionCanvas, videoResolution, currentPoints, updatePoints, redrawCanvas);
});

// Handler for Image Canvas
imgZoneCanvas.addEventListener("click", (e) => {
    handleCanvasClick(e, imgZoneCanvas, imgResolution, imgZonePoints, updateImgZonePoints, redrawImgZoneCanvas);
});

function handleCanvasClick(e, canvas, resolution, pointsArray, updateFn, redrawFn) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0 || resolution.width === 0) return;

    const scaleX = resolution.width / rect.width;
    const scaleY = resolution.height / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    pointsArray.push([x, y]);
    updateFn();
    redrawFn();
}

clearPointsBtn.addEventListener("click", () => {
    currentPoints = [];
    updatePoints();
    redrawCanvas();
});

imgZoneClearBtn.addEventListener("click", () => {
    imgZonePoints = [];
    updateImgZonePoints();
    redrawImgZoneCanvas();
});

function updatePoints() {
    pointCountSpan.textContent = currentPoints.length;
    regionPointsInput.value = "[" + currentPoints.map(p => `(${p[0]}, ${p[1]})`).join(", ") + "]";
}

function updateImgZonePoints() {
    imgZonePointCountSpan.textContent = imgZonePoints.length;
    imgZonePointsInput.value = "[" + imgZonePoints.map(p => `(${p[0]}, ${p[1]})`).join(", ") + "]";
}

function redrawCanvas() {
    drawOnCanvas(regionCanvas, currentFrameImage, currentPoints, videoResolution);
}

function redrawImgZoneCanvas() {
    drawOnCanvas(imgZoneCanvas, currentImgImage, imgZonePoints, imgResolution);
}

function drawOnCanvas(canvas, bgImage, points, resolution) {
    if (!bgImage) return;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(bgImage, 0, 0);

    if (points.length === 0) return;

    const scaleLine = Math.max(2, resolution.width / 400);
    const scaleRadius = Math.max(3, resolution.width / 200);

    // 1. Draw Lines/Polygon
    if (points.length > 1) {
        ctx.beginPath();
        ctx.lineWidth = scaleLine;
        ctx.strokeStyle = "#00b894"; // success color

        ctx.moveTo(points[0][0], points[0][1]);

        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1]);
        }

        // Close polygon if > 2 points (draw line back to start)
        if (points.length > 2) {
            ctx.lineTo(points[0][0], points[0][1]);
            // Optional: fill polygon with transparent color
            ctx.fillStyle = "rgba(0, 184, 148, 0.2)";
            ctx.fill();
        }

        ctx.stroke();
    }

    // 2. Draw Points (Vertices) on top
    ctx.fillStyle = "#ff7675"; // point color

    for (let i = 0; i < points.length; i++) {
        ctx.beginPath();
        ctx.arc(points[i][0], points[i][1], scaleRadius, 0, Math.PI * 2);
        ctx.fill();
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

        // Format the counts nicely
        if (Object.keys(data.counts).length === 0) {
            resultsEl.textContent = "No objects detected";
        } else {
            resultsEl.textContent = JSON.stringify(data.counts, null, 2);
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

        // Show download button
        downloadBtn.href = data.annotated_file;
        downloadBtn.style.display = "inline-block";

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        resultsEl.textContent = `❌ Error: ${error.message}\n\nPlease check:\n- File format is supported\n- Server is running\n- File size is reasonable`;
        resultsEl.style.color = "#d63031";
    } finally {
        // Reset button state
        btnText.style.display = "inline";
        btnLoader.style.display = "none";
        form.querySelector(".submit-btn").disabled = false;
    }
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
    if (!file) {
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
