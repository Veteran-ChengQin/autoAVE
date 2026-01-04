import json

file1_path = "/data/veteran/project/dataE/Attribute_AKS/results/exp1_local_key_frame_100/metrics_exp1_local_key_frames_threshold_0.5.json"

file2_path = "/data/veteran/project/dataE/Attribute_AKS/results/exp4_api_video_url_100/metrics_exp4_api_video_url_threshold_0.5.json"

with open(file1_path, 'r') as f:
    metrics1 = json.load(f)

with open(file2_path, 'r') as f:
    metrics2 = json.load(f)

keys1 = metrics1['per_product'].keys()
keys2 = metrics2['per_product'].keys()

# 判断key是否相等
if keys1 == keys2:
    print("key相等")
else:
    print("key不相等")