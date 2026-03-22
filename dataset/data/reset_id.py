import json

file_path = "dataset/data/outsrc.json"

with open(file_path, "r", encoding="utf-8") as f:
    problems = json.load(f)

for idx, prob in enumerate(problems):
    prob["id"] = idx
    prob["image_dir"] = ""

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(problems, f, ensure_ascii=False, indent=2)

print("finish")