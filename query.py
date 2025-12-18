from skeleton import extract_pose_from_video, add_pose_vecs_to_json
from extract_video import create_three_videos

# --------------------------
# 변수 하나만 바꾸면 됨
# --------------------------
dance_id = 7  # 숫자만 바꾸면 모든 파일명이 자동 변경됨

# --------------------------
# 경로 자동 formatting
# --------------------------
video_path = f"./content/IDEA_RUN_DEMO_{dance_id}.mp4"
json_out_path = f"./skeleton_json/{dance_id}_dance_main_pose.json"
pose_vec_json_path = f"./skeleton_json/{dance_id}_dance_main_pose_with_vecs.json"

# --------------------------
# 1) 스켈레톤 JSON 추출
# --------------------------
extract_pose_from_video(video_path, json_out_path, sample_stride=1)

# --------------------------
# 2) pose_vec 넣은 JSON 저장
# --------------------------
pose_vec_json_path = add_pose_vecs_to_json(json_out_path)

# --------------------------
# 3) 세 가지 영상 생성
# --------------------------
create_three_videos(
    video_path,
    pose_vec_json_path,
    out_original=f"./created_video/{dance_id}_original.mp4",
    out_skeleton=f"./created_video/{dance_id}_skeleton.mp4",
    out_overlay=f"./created_video/{dance_id}_overlay.mp4"
)
