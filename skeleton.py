# ============================================
# 1. Imports & 기본 설정
# ============================================
import cv2
import mediapipe as mp
import json
from tqdm import tqdm
import numpy as np
import librosa
import os

# MediaPipe Pose 준비
mp_pose = mp.solutions.pose

# MediaPipe Pose의 관절 인덱스 (참고: mediapipe 공식 문서 기준)
POSE_LANDMARKS = mp_pose.PoseLandmark

# 자주 쓸 인덱스들
IDX_LEFT_HIP = POSE_LANDMARKS.LEFT_HIP.value
IDX_RIGHT_HIP = POSE_LANDMARKS.RIGHT_HIP.value
IDX_LEFT_SHOULDER = POSE_LANDMARKS.LEFT_SHOULDER.value
IDX_RIGHT_SHOULDER = POSE_LANDMARKS.RIGHT_SHOULDER.value

NUM_JOINTS = len(POSE_LANDMARKS)  # 보통 33

# ============================================
# 2. 비디오에서 MediaPipe Pose 스켈레톤 추출
#    → JSON 파일로 저장
# ============================================

def extract_pose_from_video(
    video_path,
    json_out_path,
    sample_stride=1,
    min_det_conf=0.5,
    min_track_conf=0.5
):
    """
    video_path: 입력 비디오 경로
    json_out_path: 결과 JSON 저장 경로
    sample_stride: 프레임 간 샘플링 간격 (1이면 모든 프레임)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
    )

    results_list = []
    frame_idx = 0
    sampled_idx = 0

    pbar = tqdm(total=total_frames, desc="Extracting pose")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_stride != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        frame_data = {
            "frame_idx": int(frame_idx),
            "sample_idx": int(sampled_idx),  # 샘플링 후의 인덱스
            "time_sec": float(frame_idx / fps),
            "has_pose": False,
            "keypoints": None,  # list of {x,y,z,visibility}
        }

        if res.pose_landmarks:
            frame_data["has_pose"] = True
            keypoints = []
            for lm in res.pose_landmarks.landmark:
                keypoints.append({
                    "x": float(lm.x),           # 0~1 (width 기준)
                    "y": float(lm.y),           # 0~1 (height 기준)
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                })
            frame_data["keypoints"] = keypoints

        results_list.append(frame_data)
        frame_idx += 1
        sampled_idx += 1
        pbar.update(1)

    cap.release()
    pose.close()
    pbar.close()

    data = {
        "meta": {
            "video_path": video_path,
            "fps": float(fps),
            "num_frames_raw": int(total_frames),
            "num_frames_sampled": int(len(results_list)),
            "sample_stride": int(sample_stride),
            "pose_model": "mediapipe_pose",
            "num_joints": int(NUM_JOINTS),
        },
        "frames": results_list
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Pose JSON saved to: {json_out_path}")
    return json_out_path
# ============================================
# 3-1. 2x2 회전 행렬 생성
# ============================================
def rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float32)
# ============================================
# 3-2. keypoints (dict list) -> 상대 포즈 벡터 (flatten)
#     - root: 좌/우 엉덩이 중간점
#     - scale: 양 어깨 거리
#     - orientation: 엉덩이 벡터를 x축에 align
# ============================================
def keypoints_to_relative_pose_vec(
    keypoints,
    visibility_thresh=0.5,
    use_rotation=True
):
    """
    keypoints: list of dicts [{x,y,z,visibility}, ...], length=NUM_JOINTS

    return:
        pose_vec: (NUM_JOINTS*2,) numpy array, dtype float32
                  [x0,y0,x1,y1,...] 정규화된 2D 상대 좌표
        valid_mask: (NUM_JOINTS,) visibility 기반 유효 여부 (True/False)
    """
    if keypoints is None or len(keypoints) != NUM_JOINTS:
        return None, None

    kp = np.array([[k["x"], k["y"]] for k in keypoints], dtype=np.float32)  # (J,2)
    visibility = np.array([k["visibility"] for k in keypoints], dtype=np.float32)
    valid_mask = visibility > visibility_thresh

    # --- 1) root: 좌/우 엉덩이 중간점 ---
    left_hip = kp[IDX_LEFT_HIP]
    right_hip = kp[IDX_RIGHT_HIP]
    root = (left_hip + right_hip) / 2.0

    # translation 제거
    kp_rel = kp - root  # (J,2)

    # --- 2) scale: 양 어깨 거리 ---
    left_shoulder = kp[IDX_LEFT_SHOULDER]
    right_shoulder = kp[IDX_RIGHT_SHOULDER]
    shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder) + 1e-8
    kp_rel /= shoulder_dist  # 스케일 정규화

    # --- 3) orientation: 엉덩이 방향을 x축에 align ---
    if use_rotation:
        hip_vec = right_hip - left_hip
        angle = np.arctan2(hip_vec[1], hip_vec[0])  # 현재 x축과의 각도
        R = rotation_matrix(-angle)  # 반대 방향 회전
        kp_rel = (R @ kp_rel.T).T  # (2,J) -> (J,2)

    pose_vec = kp_rel.reshape(-1).astype(np.float32)  # (J*2,)
    return pose_vec, valid_mask

# ============================================
# 3-3. JSON 전체에 pose_vec 추가
# ============================================
def add_pose_vecs_to_json(json_path, out_json_path=None):
    """
    json_path: extract_pose_from_video로 만든 JSON
    out_json_path: 저장할 경로. None이면 원본 이름 뒤에 _with_vecs.json
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    new_frames = []

    for fdata in tqdm(frames, desc="Computing pose_vec"):
        if fdata.get("has_pose") and fdata.get("keypoints") is not None:
            pose_vec, valid_mask = keypoints_to_relative_pose_vec(fdata["keypoints"])
            if pose_vec is None:
                fdata["has_pose"] = False
                fdata["pose_vec"] = None
                fdata["valid_mask"] = None
            else:
                fdata["pose_vec"] = pose_vec.tolist()
                fdata["valid_mask"] = valid_mask.astype(np.uint8).tolist()
        else:
            fdata["pose_vec"] = None
            fdata["valid_mask"] = None

        new_frames.append(fdata)

    data["frames"] = new_frames

    if out_json_path is None:
        root, ext = os.path.splitext(json_path)
        out_json_path = root + "_with_vecs.json"

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] pose_vec added & saved to: {out_json_path}")
    return out_json_path

video_path = "./content/IDEA_RUN_DEMO_1.mp4"    # 업로드한 영상
json_out_path = "./content/dance_main_pose.json"

extract_pose_from_video(video_path, json_out_path, sample_stride=1)

pose_json_path = "./content/dance_main_pose.json"
pose_vec_json_path = add_pose_vecs_to_json(pose_json_path)
