function uploadImage() {
    let fileInput = document.getElementById("imageUpload");
    if (!fileInput.files.length) {
        alert("Please select an image!");
        return;
    }

    let formData = new FormData();
    formData.append("image", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        let resultDiv = document.getElementById("result");
        resultDiv.innerHTML = `<strong>Predicted Stage:</strong> ${data.predicted_class}<br>`;

        for (let [stage, confidence] of Object.entries(data.confidence_scores)) {
            resultDiv.innerHTML += `${stage}: ${confidence.toFixed(2)}%<br>`;
        }

        document.getElementById("spmImage").src = "data:image/png;base64," + data.spm_image;
        document.getElementById("spmContainer").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
}
