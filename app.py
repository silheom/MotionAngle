from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import os
import uuid

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "app": "MotionAngle"})

@app.post("/analyze")
def analyze():
    # Stage 1 placeholder:
    # The upload pipeline is wired first so the app can be deployed and tested.
    # Pose estimation will be added after the basic Render deployment succeeds.
    video = request.files.get("video")
    x_value = request.form.get("x", "5")

    if video is None or not video.filename:
        return jsonify({"error": "영상 파일이 없습니다."}), 400

    try:
        x_value = max(1, min(10, int(x_value)))
    except ValueError:
        x_value = 5

    ext = Path(video.filename).suffix.lower() or ".mp4"
    file_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{file_id}{ext}"
    video.save(saved_path)

    return jsonify({
        "ok": True,
        "message": "영상 업로드가 완료되었습니다. 다음 단계에서 AI 포즈 분석을 연결합니다.",
        "x": x_value,
        "file_id": file_id
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
