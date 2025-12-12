const form = document.getElementById("uploadForm");
const resultsEl = document.getElementById("results");
const imgEl = document.getElementById("annotatedImage");
const videoEl = document.getElementById("annotatedVideo");
const typeSelect = document.getElementById("typeSelect");
const slicedImageSettings = document.getElementById("slicedImageSettings");
const slicedVideoSettings = document.getElementById("slicedVideoSettings");

// Show/hide settings based on processing type
typeSelect.addEventListener("change", () => {
    // Hide all settings first
    slicedImageSettings.style.display = "none";
    slicedVideoSettings.style.display = "none";

    // Show appropriate settings
    if (typeSelect.value === "sliced") {
        slicedImageSettings.style.display = "block";
    } else if (typeSelect.value === "sliced_video") {
        slicedVideoSettings.style.display = "block";
    }
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const model = form.model.value;
    const conf = form.conf_threshold.value;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("model", model);
    formData.append("conf_threshold", conf);

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

    let endpoint = "/basic-count";
    switch (typeSelect.value) {
        case "basic": endpoint = "/basic-count"; break;
        case "sliced": endpoint = "/sliced-count"; break;
        case "video": endpoint = "/video-count"; break;
        case "sliced_video": endpoint = "/sliced-video-count"; break;
    }

    resultsEl.textContent = "Processing...";
    imgEl.style.display = "none";
    videoEl.style.display = "none";

    const response = await fetch(endpoint, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        resultsEl.textContent = `Error: ${response.status} ${response.statusText}`;
        return;
    }

    const data = await response.json();
    resultsEl.textContent = JSON.stringify(data.counts, null, 2);

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
});
