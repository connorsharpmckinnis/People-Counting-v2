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
    }
});

// Interactive Region Drawing Logic
const regionCanvas = document.getElementById("regionCanvas");
const canvasPlaceholder = document.getElementById("canvasPlaceholder");
const regionPointsInput = document.getElementById("regionPoints");
const clearPointsBtn = document.getElementById("clearPointsBtn");
const pointCountSpan = document.getElementById("pointCount");
let currentPoints = [];
let videoResolution = { width: 0, height: 0 };
let currentFrameImage = null; // Store the frame for redrawing

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0 && e.target.files[0].type.startsWith("video/")) {
        loadVideoFrame(e.target.files[0]);
    } else {
        resetCanvas();
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

function resetCanvas() {
    regionCanvas.style.display = "none";
    canvasPlaceholder.style.display = "block";
    currentPoints = [];
    updatePoints();
}

regionCanvas.addEventListener("click", (e) => {
    const rect = regionCanvas.getBoundingClientRect();

    // Check if we have valid dimensions
    if (rect.width === 0 || rect.height === 0 || videoResolution.width === 0) return;

    // Calculate scaling factors (Displayed Size vs Actual Video Size)
    const scaleX = videoResolution.width / rect.width;
    const scaleY = videoResolution.height / rect.height;

    // Get click coordinates relative to the video resolution
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    currentPoints.push([x, y]);
    updatePoints();
    redrawCanvas();
});

clearPointsBtn.addEventListener("click", () => {
    currentPoints = [];
    updatePoints();
    redrawCanvas();
});

function updatePoints() {
    pointCountSpan.textContent = currentPoints.length;

    // Format as Python list of tuples: [(x, y), (x, y)]
    const tupleString = "[" + currentPoints.map(p => `(${p[0]}, ${p[1]})`).join(", ") + "]";
    regionPointsInput.value = tupleString;
}

function redrawCanvas() {
    if (!currentFrameImage) return;

    const ctx = regionCanvas.getContext("2d");

    // Restore original frame
    ctx.putImageData(currentFrameImage, 0, 0);

    if (currentPoints.length === 0) return;

    const scaleLine = Math.max(2, videoResolution.width / 400);
    const scaleRadius = Math.max(3, videoResolution.width / 200);

    // 1. Draw Lines/Polygon
    if (currentPoints.length > 1) {
        ctx.beginPath();
        ctx.lineWidth = scaleLine;
        ctx.strokeStyle = "#00b894"; // success color

        ctx.moveTo(currentPoints[0][0], currentPoints[0][1]);

        for (let i = 1; i < currentPoints.length; i++) {
            ctx.lineTo(currentPoints[i][0], currentPoints[i][1]);
        }

        // Close polygon if > 2 points (draw line back to start)
        if (currentPoints.length > 2) {
            ctx.lineTo(currentPoints[0][0], currentPoints[0][1]);
            // Optional: fill polygon with transparent color
            ctx.fillStyle = "rgba(0, 184, 148, 0.2)";
            ctx.fill();
        }

        ctx.stroke();
    }

    // 2. Draw Points (Vertices) on top
    ctx.fillStyle = "#ff7675"; // point color

    for (let i = 0; i < currentPoints.length; i++) {
        ctx.beginPath();
        ctx.arc(currentPoints[i][0], currentPoints[i][1], scaleRadius, 0, Math.PI * 2);
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

    let endpoint = "/basic-count";
    switch (typeSelect.value) {
        case "basic": endpoint = "/basic-count"; break;
        case "sliced": endpoint = "/sliced-count"; break;
        case "video": endpoint = "/video-count"; break;
        case "sliced_video": endpoint = "/sliced-video-count"; break;
        case "polygon_cross_count": endpoint = "/polygon-cross-count"; break;
    }

    resultsEl.textContent = "Processing your file...";
    imgEl.style.display = "none";
    videoEl.style.display = "none";
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
