from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import os
import uuid

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# -------------------------
# Frontend
# -------------------------

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")


@app.get("/script.js")
def script():
    return send_from_directory(BASE_DIR, "script.js")


# -------------------------
# Health check
# -------------------------

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": "MotionAngle"
    })


# -------------------------
# Video analysis endpoint
# -------------------------

@app.post("/analyze")
def analyze():
    video = request.files.get("video")
    x_value = request.form.get("x", "5")

    if video is None or not video.filename:
        return jsonify({
            "error": "영상 파일이 없습니다."
        }), 400

    try:
        x_value = int(x_value)
        x_value = max(1, min(10, x_value))
    except (ValueError, TypeError):
        x_value = 5

    # 업로드된 영상의 확장자 유지
    ext = Path(video.filename).suffix.lower()

    if not ext:
        ext = ".mp4"

    # 임시 고유 파일명 생성
    file_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        video.save(saved_path)
    except Exception as e:
        return jsonify({
            "error": "영상 저장에 실패했습니다.",
            "detail": str(e)
        }), 500

    return jsonify({
        "ok": True,
        "message": "영상 업로드가 완료되었습니다.",
        "x": x_value,
        "file_id": file_id
    })


# -------------------------
# Server
# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
