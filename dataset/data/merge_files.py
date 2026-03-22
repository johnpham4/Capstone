import json

files = ["dataset/data/havamath8.json", "dataset/data/minh_data.json"]

problems = []
global_idx = 0

for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        probs = json.load(f)

    for prob in probs:
        problem = {}
        problem["id"] = global_idx
        problem["image_dir"] = ""
        problem["problem"] = prob.get("problem", "")
        problems.append(problem)

        global_idx += 1

with open("dataset/data/outsrc.json", "w", encoding="utf-8") as f:
    json.dump(problems, f, ensure_ascii=False, indent=2)

print("finish")