import sys
from datasets import load_dataset

split = sys.argv[1] if len(sys.argv) > 1 else "test"
ds = load_dataset("quangne/CGL-Text2Geo", split=split)
out = [f"{split} rows: {len(ds)}"]
for i in range(5):
    row = ds[i]
    out.append(f"--- row {i} (id={row['id']}) ---")
    out.append("instruction: " + repr(row["instruction"][:150]))
    out.append("answer: " + repr(row["answer"][:400]))
    out.append("")
with open("profiling/_ds_sample.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("ok")
