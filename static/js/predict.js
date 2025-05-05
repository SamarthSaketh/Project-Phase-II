async function predictDisease() {
    const fileInput = document.getElementById("imageUpload");
    const resultDiv = document.getElementById("result");

    if (!fileInput.files.length) {
        resultDiv.innerHTML = "<p style='color: red;'>Please select an image!</p>";
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onloadend = async function () {
        const base64Image = reader.result.split(',')[1];  // Extract Base64 Data

        resultDiv.innerHTML = "<p>Predicting...</p>";  // Show loading message

        try {
            const response = await fetch("http://localhost:5001/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: base64Image }),
            });

            const data = await response.json();

            if (data.error) {
                resultDiv.innerHTML = `<p style='color: red;'>Error: ${data.error}</p>`;
            } else {
                resultDiv.innerHTML = `
                    <h3>Prediction Result</h3>
                    <p><strong>Disease:</strong> ${data.prediction}</p>
                    <p><strong>Confidence:</strong> ${data.confidence}%</p>
                `;
            }
        } catch (error) {
            resultDiv.innerHTML = "<p style='color: red;'>Error connecting to server.</p>";
        }
    };

    reader.readAsDataURL(file);
}
app.post('/upload', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }
    res.json({ message: "File uploaded successfully", filename: req.file.filename });
});
