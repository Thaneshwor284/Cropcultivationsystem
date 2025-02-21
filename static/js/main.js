document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const fileInput = document.getElementById('file-input');
    const uploadZone = document.getElementById('upload-zone');
    const previewContainerUpload = document.getElementById('preview-container-upload');
    const previewImage = document.getElementById('preview-image');
    const uploadResult = document.getElementById('upload-result');
    const uploadResultText = document.getElementById('upload-result-text');
    
    const startCameraBtn = document.getElementById('start-camera');
    const capturePhotoBtn = document.getElementById('capture-photo');
    const cameraFeed = document.getElementById('camera-feed');
    const capturedPhoto = document.getElementById('captured-photo');
    const cameraResult = document.getElementById('camera-result');
    const cameraResultText = document.getElementById('camera-result-text');

    // Sensor elements
    const temperatureValue = document.getElementById('temperature-value');
    const humidityValue = document.getElementById('humidity-value');
    const soilMoistureValue = document.getElementById('soil-moisture-value');
    const alertsContainer = document.getElementById('alerts-container');
    const lastUpdated = document.getElementById('last-updated');

    let stream = null;
    let sensorUpdateInterval = null;

    // Start sensor data updates
    function startSensorUpdates() {
        // Update immediately
        updateSensorData();
        // Then update every 1 minute (60000 milliseconds)
        sensorUpdateInterval = setInterval(updateSensorData, 60000);
    }

    async function updateSensorData() {
        try {
            const response = await fetch('/sensor_data');
            const result = await response.json();
            
            if (result.success) {
                // Update sensor values
                temperatureValue.textContent = `${result.data.temperature}°C`;
                humidityValue.textContent = `${result.data.humidity}%`;
                soilMoistureValue.textContent = `${result.data.soil_moisture}%`;
                lastUpdated.textContent = `Last updated: ${result.timestamp}`;

                // Update sensor gauges
                updateGauge('temperature-gauge', result.data.temperature, 0, 40);
                updateGauge('humidity-gauge', result.data.humidity, 0, 100);
                updateGauge('soil-moisture-gauge', result.data.soil_moisture, 0, 100);

                // Display alerts
                displayAlerts(result.alerts);
            }
        } catch (err) {
            console.error('Error updating sensor data:', err);
        }
    }

    function updateGauge(gaugeId, value, min, max) {
        const gauge = document.getElementById(gaugeId);
        if (gauge) {
            const percentage = ((value - min) / (max - min)) * 100;
            gauge.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
            
            // Update gauge color based on value
            if (percentage > 80) {
                gauge.className = 'gauge-fill danger';
            } else if (percentage < 20) {
                gauge.className = 'gauge-fill warning';
            } else {
                gauge.className = 'gauge-fill normal';
            }
        }
    }

    function displayAlerts(alerts) {
        alertsContainer.innerHTML = '';
        alerts.forEach(alert => {
            const alertElement = document.createElement('div');
            alertElement.className = `alert alert-${alert.type} d-flex align-items-center alert-dismissible fade show`;
            alertElement.innerHTML = `
                <div class="alert-icon me-3">
                    <i class="fas ${alert.type === 'danger' ? 'fa-exclamation-circle' : 'fa-exclamation-triangle'}"></i>
                </div>
                <div>
                    <strong>${alert.sensor.replace('_', ' ').toUpperCase()}:</strong> ${alert.message}
                    <div class="small mt-1">Current value: ${alert.value}${alert.unit}</div>
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            alertsContainer.appendChild(alertElement);
        });
    }

    // Start sensor updates when page loads
    startSensorUpdates();

    // Cleanup intervals when leaving page
    window.addEventListener('beforeunload', () => {
        if (sensorUpdateInterval) {
            clearInterval(sensorUpdateInterval);
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });

    // File Upload Handling
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.backgroundColor = 'rgba(46, 204, 113, 0.2)';
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.backgroundColor = 'rgba(46, 204, 113, 0.05)';
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.backgroundColor = 'rgba(46, 204, 113, 0.05)';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });

    async function handleFile(file) {
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainerUpload.style.display = 'block';
        };
        reader.readAsDataURL(file);

        // Upload and get prediction
        try {
            uploadResult.style.display = 'block';
            uploadResult.className = 'result-card';
            uploadResultText.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            uploadResult.querySelector('.confidence-level').style.width = '0%';

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                uploadResult.className = 'result-card success';
                if (result.human_detected) {
                    uploadResultText.innerHTML = `
                        <i class="fas fa-user me-2"></i>
                        <div>
                            <strong>Human Detected</strong>
                            <div class="mt-2">${result.message}</div>
                        </div>
                    `;
                    uploadResult.querySelector('.confidence-level').style.width = '100%';
                } else {
                    const confidence = parseFloat(result.confidence);
                    uploadResultText.innerHTML = `
                        <i class="fas fa-check-circle me-2"></i>
                        <div>
                            <strong>Detection Result:</strong>
                            <div class="disease-badge">${result.disease}</div>
                            <div class="mt-2">Confidence: ${confidence.toFixed(1)}%</div>
                        </div>
                    `;
                    uploadResult.querySelector('.confidence-level').style.width = `${confidence}%`;
                }
            } else {
                uploadResult.className = 'result-card error';
                uploadResultText.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>${result.error}`;
                uploadResult.querySelector('.confidence-level').style.width = '0%';
            }
        } catch (err) {
            console.error('Error uploading file:', err);
            uploadResult.className = 'result-card error';
            uploadResultText.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Error processing image';
            uploadResult.querySelector('.confidence-level').style.width = '0%';
        }
    }

    // Camera Handling
    startCameraBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: {
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            });
            cameraFeed.srcObject = stream;
            cameraFeed.style.display = 'block';
            startCameraBtn.style.display = 'none';
            capturePhotoBtn.style.display = 'inline-block';
            capturedPhoto.style.display = 'none';
            cameraResult.style.display = 'none';
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Error accessing camera. Please make sure you have granted camera permissions.');
        }
    });

    capturePhotoBtn.addEventListener('click', async () => {
        if (!stream) return;

        // Capture the current frame
        const canvas = capturedPhoto;
        canvas.width = cameraFeed.videoWidth;
        canvas.height = cameraFeed.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraFeed, 0, 0);

        // Hide video and show canvas
        cameraFeed.style.display = 'none';
        canvas.style.display = 'block';
        
        // Stop the camera
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        
        // Show start camera button again
        startCameraBtn.style.display = 'inline-block';
        capturePhotoBtn.style.display = 'none';

        // Get prediction
        try {
            cameraResult.style.display = 'block';
            cameraResult.className = 'result-card';
            cameraResultText.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            cameraResult.querySelector('.confidence-level').style.width = '0%';

            const imageData = canvas.toDataURL('image/jpeg');
            const response = await fetch('/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });

            const result = await response.json();
            
            if (result.success) {
                cameraResult.className = 'result-card success';
                if (result.human_detected) {
                    cameraResultText.innerHTML = `
                        <i class="fas fa-user me-2"></i>
                        <div>
                            <strong>Human Detected</strong>
                            <div class="mt-2">${result.message}</div>
                        </div>
                    `;
                    cameraResult.querySelector('.confidence-level').style.width = '100%';
                } else {
                    const confidence = parseFloat(result.confidence);
                    cameraResultText.innerHTML = `
                        <i class="fas fa-check-circle me-2"></i>
                        <div>
                            <strong>Detection Result:</strong>
                            <div class="disease-badge">${result.disease}</div>
                            <div class="mt-2">Confidence: ${confidence.toFixed(1)}%</div>
                        </div>
                    `;
                    cameraResult.querySelector('.confidence-level').style.width = `${confidence}%`;
                }
            } else {
                cameraResult.className = 'result-card error';
                cameraResultText.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>${result.error}`;
                cameraResult.querySelector('.confidence-level').style.width = '0%';
            }
        } catch (err) {
            console.error('Error during detection:', err);
            cameraResult.className = 'result-card error';
            cameraResultText.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Error processing image';
            cameraResult.querySelector('.confidence-level').style.width = '0%';
        }
    });
});
