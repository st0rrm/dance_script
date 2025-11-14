from skeleton import extract_pose_from_video, add_pose_vecs_to_json
from extract_video import create_three_videos



video_path = "./content/IDEA_RUN_DEMO_4.mp4"    # 업로드한 영상
json_out_path = "./skeleton_json/4_dance_main_pose.json"

extract_pose_from_video(video_path, json_out_path, sample_stride=1)

pose_json_path = json_out_path
pose_vec_json_path = add_pose_vecs_to_json(pose_json_path)


# 이 코드로 영상, 스켈레톤, 영상 + 스켈레톤이 가능

pose_json = "./skeleton_json/4_dance_main_pose_with_vecs.json"

create_three_videos(video_path, pose_json,
    out_original="./created_video/4_original.mp4",
    out_skeleton="./created_video/4_skeleton.mp4",
    out_overlay="./created_video/4_overlay.mp4"
)