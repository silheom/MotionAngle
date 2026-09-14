from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import os
import math
import tempfile
import traceback

import cv2
import mediapipe as mp


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")


@app.get("/script.js")
def script():
    return send_from_directory(BASE_DIR, "script.js")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": "MotionAngle",
        "mediapipe": mp.__version__
    })


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None

    ax, ay = a
    bx, by = b
    cx, cy = c

    ba_x = ax - bx
    ba_y = ay - by
    bc_x = cx - bx
    bc_y = cy - by

    length_ba = math.hypot(ba_x, ba_y)
    length_bc = math.hypot(bc_x, bc_y)

    if length_ba < 1e-8 or length_bc < 1e-8:
        return None

    cosine = (
        (ba_x * bc_x + ba_y * bc_y)
        / (length_ba * length_bc)
    )

    cosine = clamp(cosine, -1.0, 1.0)

    return round(
        math.degrees(math.acos(cosine)),
        1
    )


POSE_POINTS = {
    "nose": mp.solutions.pose.PoseLandmark.NOSE,
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "left_index": mp.solutions.pose.PoseLandmark.LEFT_INDEX,
    "right_index": mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
    "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    "left_foot": mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
    "right_foot": mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
}


ANGLE_DEFINITIONS = {
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_wrist": ("left_elbow", "left_wrist", "left_index"),
    "right_wrist": ("right_elbow", "right_wrist", "right_index"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_ankle": ("left_knee", "left_ankle", "left_foot"),
    "right_ankle": ("right_knee", "right_ankle", "right_foot"),
}


BODY_SEGMENTS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
]


def extract_pose_points(results):
    if results is None or results.pose_landmarks is None:
        return None

    landmarks = results.pose_landmarks.landmark
    points = {}

    for name, landmark_id in POSE_POINTS.items():
        landmark = landmarks[landmark_id.value]
        visibility = float(landmark.visibility)

        if visibility < 0.5:
            points[name] = None
        else:
            points[name] = {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "visibility": visibility
            }

    return points


def interpolate_joint_series(values):
    result = list(values)

    valid_indices = [
        i for i, value in enumerate(result)
        if value is not None
    ]

    if not valid_indices:
        return result

    first = valid_indices[0]

    for i in range(0, first):
        result[i] = result[first]

    last = valid_indices[-1]

    for i in range(last + 1, len(result)):
        result[i] = result[last]

    for start_index, end_index in zip(
        valid_indices,
        valid_indices[1:]
    ):
        if end_index - start_index <= 1:
            continue

        start_value = result[start_index]
        end_value = result[end_index]
        gap = end_index - start_index

        for i in range(start_index + 1, end_index):
            ratio = (i - start_index) / gap

            result[i] = {
                "x": (
                    start_value["x"]
                    + (end_value["x"] - start_value["x"]) * ratio
                ),
                "y": (
                    start_value["y"]
                    + (end_value["y"] - start_value["y"]) * ratio
                ),
                "visibility": min(
                    start_value.get("visibility", 1.0),
                    end_value.get("visibility", 1.0)
                )
            }

    return result


def interpolate_all_joints(sampled_frames, frame_count):
    all_points = {}

    for joint_name in POSE_POINTS:
        values = []

        for frame_index in range(frame_count):
            frame_points = sampled_frames.get(frame_index)

            if frame_points is None:
                values.append(None)
            else:
                values.append(frame_points.get(joint_name))

        all_points[joint_name] = interpolate_joint_series(values)

    return all_points


def calculate_all_angles(all_points, frame_count):
    angles = {
        joint_name: [None] * frame_count
        for joint_name in ANGLE_DEFINITIONS
    }

    for joint_name, definition in ANGLE_DEFINITIONS.items():
        point_a_series = all_points[definition[0]]
        point_b_series = all_points[definition[1]]
        point_c_series = all_points[definition[2]]

        for frame_index in range(frame_count):
            point_a = point_a_series[frame_index]
            point_b = point_b_series[frame_index]
            point_c = point_c_series[frame_index]

            if point_a is None or point_b is None or point_c is None:
                continue

            angles[joint_name][frame_index] = calculate_angle(
                (point_a["x"], point_a["y"]),
                (point_b["x"], point_b["y"]),
                (point_c["x"], point_c["y"])
            )

    return angles


def remove_angle_outliers(values, threshold=35.0):
    if len(values) < 3:
        return values[:]

    result = values[:]

    for i in range(1, len(values) - 1):
        current = values[i]

        if current is None:
            continue

        nearby = [
            values[i - 1],
            values[i],
            values[i + 1]
        ]

        nearby = [
            value for value in nearby
            if value is not None
        ]

        if len(nearby) < 2:
            continue

        median = sorted(nearby)[len(nearby) // 2]

        if abs(current - median) > threshold:
            result[i] = median

    return result


def clean_all_angles(angles):
    return {
        joint_name: remove_angle_outliers(values)
        for joint_name, values in angles.items()
    }


def calculate_angle_statistics(angles, fps):
    statistics = {}

    for joint_name, values in angles.items():
        valid_values = [
            (index, value)
            for index, value in enumerate(values)
            if value is not None
        ]

        if not valid_values:
            statistics[joint_name] = {
                "max": None,
                "min": None,
                "rom": None,
                "max_frame": None,
                "min_frame": None,
                "max_time": None,
                "min_time": None
            }
            continue

        max_frame, max_value = max(
            valid_values,
            key=lambda item: item[1]
        )

        min_frame, min_value = min(
            valid_values,
            key=lambda item: item[1]
        )

        statistics[joint_name] = {
            "max": round(float(max_value), 1),
            "min": round(float(min_value), 1),
            "rom": round(float(max_value - min_value), 1),
            "max_frame": int(max_frame),
            "min_frame": int(min_frame),
            "max_time": round(max_frame / fps, 3),
            "min_time": round(min_frame / fps, 3)
        }

    return statistics


def build_frame_results(all_points, angles, frame_count, fps):
    frame_results = []

    for frame_index in range(frame_count):
        frame_points = {}

        for joint_name in POSE_POINTS:
            point = all_points[joint_name][frame_index]

            if point is None:
                frame_points[joint_name] = None
            else:
                frame_points[joint_name] = [
                    round(float(point["x"]), 5),
                    round(float(point["y"]), 5)
                ]

        frame_angles = {}

        for joint_name in ANGLE_DEFINITIONS:
            value = angles[joint_name][frame_index]

            if value is None:
                frame_angles[joint_name] = None
            else:
                frame_angles[joint_name] = round(float(value), 1)

        frame_results.append({
            "frame": frame_index,
            "time": round(frame_index / fps, 3),
            "points": frame_points,
            "angles": frame_angles
        })

    return frame_results


def analyze_video_path(video_path, x_value):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    reported_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    sampled_frames = {}
    actual_frame_count = 0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_index = actual_frame_count
            actual_frame_count += 1

            if frame_index % x_value != 0:
                continue

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = pose.process(rgb_frame)

            sampled_frames[frame_index] = extract_pose_points(
                results
            )

    finally:
        pose.close()
        cap.release()

    if actual_frame_count == 0:
        raise RuntimeError(
            "영상에서 프레임을 읽지 못했습니다."
        )

    frame_count = actual_frame_count

    all_points = interpolate_all_joints(
        sampled_frames,
        frame_count
    )

    angles = calculate_all_angles(
        all_points,
        frame_count
    )

    angles = clean_all_angles(angles)

    return {
        "fps": round(float(fps), 3),
        "frame_count": frame_count,
        "reported_frame_count": reported_frame_count,
        "sampled_frame_count": len(sampled_frames),
        "points": all_points,
        "angles": angles
    }


@app.post("/analyze")
def analyze():
    video = request.files.get("video")
    x_value = request.form.get("x", "5")

    if video is None or not video.filename:
        return jsonify({
            "ok": False,
            "error": "영상 파일이 없습니다."
        }), 400

    try:
        x_value = int(x_value)
    except (ValueError, TypeError):
        x_value = 5

    x_value = max(1, min(10, x_value))

    extension = Path(video.filename).suffix.lower()

    if not extension:
        extension = ".mp4"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:
            temp_path = temp_file.name

        video.save(temp_path)

        analysis = analyze_video_path(
            temp_path,
            x_value
        )

        fps = analysis["fps"]
        frame_count = analysis["frame_count"]

        statistics = calculate_angle_statistics(
            analysis["angles"],
            fps
        )

        frame_results = build_frame_results(
            analysis["points"],
            analysis["angles"],
            frame_count,
            fps
        )

        duration = frame_count / fps if fps > 0 else 0.0

        return jsonify({
            "ok": True,
            "message": "AI 영상 분석이 완료되었습니다.",
            "x": x_value,
            "fps": fps,
            "frame_count": frame_count,
            "duration": round(duration, 3),
            "sampled_frame_count": analysis["sampled_frame_count"],
            "statistics": statistics,
            "segments": BODY_SEGMENTS,
            "frames": frame_results
        })

    except Exception as error:
        print(
            "영상 분석 오류:",
            traceback.format_exc()
        )

        return jsonify({
            "ok": False,
            "error": "영상 분석 중 오류가 발생했습니다.",
            "detail": str(error)
        }), 500

    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
