const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);  // nền trắng

ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

let drawing = false;

canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top); // Thêm dòng này
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

function draw(e) {
    if (!drawing) return;

    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);  // nền trắng lại
    ctx.beginPath();
    document.getElementById("result").innerText = "Kết quả: ";
}

function predict() {
    const image = canvas.toDataURL("image/png");

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: image }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText =
            `Kết quả: ${data.digit} (Độ tin cậy: ${data.confidence})`;
    });
}
