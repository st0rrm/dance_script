import cv2
import mediapipe as mp
import numpy as np
import json
from tqdm import tqdm

mp_pose = mp.solutions.pose

# Mediapipe pose connections
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# 관절 이름 (색깔, 굵기 등 설정할 때 쓸 수 있음)
def draw_skeleton(image, keypoints, visibility_thresh=0.5):
    """
    image: BGR image (H,W,3)
    keypoints: length=33, list of {x, y, z, visibility}
    """
    H, W = image.shape[:2]

    # 점 그리기
    for kp in keypoints:
        if kp["visibility"] < visibility_thresh:
            continue
        cx = int(kp["x"] * W)
        cy = int(kp["y"] * H)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)

    # 관절 연결선 그리기
    for con in POSE_CONNECTIONS:
        i1, i2 = con
        kp1 = keypoints[i1]
        kp2 = keypoints[i2]
        if kp1["visibility"] < visibility_thresh or kp2["visibility"] < visibility_thresh:
            continue

        x1, y1 = int(kp1["x"] * W), int(kp1["y"] * H)
        x2, y2 = int(kp2["x"] * W), int(kp2["y"] * H)

        cv2.line(image, (x1, y1), (x2, y2), (0, 200, 255), 2)

    return image

def create_three_videos(
    video_path,
    pose_json_path,
    out_original="output_original.mp4",
    out_skeleton="output_skeleton.mp4",
    out_overlay="output_overlay.mp4"
):
    # 원본 영상
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # JSON 로드
    with open(pose_json_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)
    frames = pose_data["frames"]

    # output video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_orig = cv2.VideoWriter(out_original, fourcc, fps, (W, H))
    vw_skel = cv2.VideoWriter(out_skeleton, fourcc, fps, (W, H))
    vw_over = cv2.VideoWriter(out_overlay, fourcc, fps, (W, H))

    print("Generating videos...")
    idx = 0

    for frame_info in tqdm(frames, total=len(frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # 원본 저장
        vw_orig.write(frame.copy())

        # 스켈레톤만 영상 생성
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        if frame_info["has_pose"] and frame_info["keypoints"] is not None:
            skel_frame = draw_skeleton(blank, frame_info["keypoints"])
        else:
            skel_frame = blank
        vw_skel.write(skel_frame)

        # 오버레이 영상
        over_frame = frame.copy()
        if frame_info["has_pose"] and frame_info["keypoints"] is not None:
            over_frame = draw_skeleton(over_frame, frame_info["keypoints"])
        vw_over.write(over_frame)

        idx += 1

    cap.release()
    vw_orig.release()
    vw_skel.release()
    vw_over.release()

    print("[OK] All videos generated!")
    print(f"- Original: {out_original}")
    print(f"- Skeleton Only: {out_skeleton}")
    print(f"- Overlay: {out_overlay}")


# 이 코드로 영상, 스켈레톤, 영상 + 스켈레톤이 가능

video_path = "./content/IDEA_RUN_DEMO_1.mp4"
pose_json = "./content/dance_main_pose_with_vecs.json"

create_three_videos(video_path, pose_json,
    out_original="video_original.mp4",
    out_skeleton="video_skeleton.mp4",
    out_overlay="video_overlay.mp4"
)
