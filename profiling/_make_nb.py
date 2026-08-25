import base64
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
repo_url = "https://github.com/johnpham4/GeoSystem.git"
patch_b64 = base64.b64encode(
    (root / "profiling" / "colab_bundle.patch").read_bytes()
).decode("ascii")

SETUP = f'''#@title 1. Clone repo + áp dụng patch (optimizer + ablation script)
import base64, pathlib

REPO = "{repo_url}"

if not pathlib.Path("/content/GeoSystem").exists():
    !git clone --depth 1 {repo_url} /content/GeoSystem

PATCH_B64 = "{patch_b64}"
%cd /content/GeoSystem
pathlib.Path("colab_bundle.patch").write_bytes(base64.b64decode(PATCH_B64))
!git apply colab_bundle.patch && echo "PATCH OK"
'''

INSTALL = '''#@title 2. Cài dependencies (~30s; torch/numpy đã có sẵn trên Colab)
!pip -q install loguru datasets
'''

SMOKE = '''#@title 3. Smoke test nhanh (~3 phút) — kiểm tra mọi thứ chạy được trước khi chạy full
!python -u profiling/ablation_initializer.py \\
  --repo quangne/CGL-Text2Geo --split test --field answer \\
  --max-samples 6 --seeds 1 --epochs 1000 --workers 2 \\
  --output profiling/colab_smoke.json
'''

FULL = '''#@title 4. CHẠY FULL ABLATION — 381 DSL x 2 modes x 5 seeds (~3 giờ trên Colab free)
#@markdown Có `--resume`: nếu Colab ngắt giữa chừng, **chạy lại cell này** — nó tiếp tục từ chỗ dừng.
!python -u profiling/ablation_initializer.py \\
  --repo quangne/CGL-Text2Geo --split test --field answer \\
  --seeds 5 --epochs 1000 --workers 2 \\
  --output profiling/ablation_cgl_test.json
'''

REPORT = '''#@title 5. Xem kết quả + bảng LaTeX (chạy sau khi cell 4 xong)
import json

with open("profiling/ablation_cgl_test.json", encoding="utf-8") as f:
    payload = json.load(f)

print(f"Elapsed: {payload['elapsed_seconds']:.0f}s\\n")
for mode, s in payload["summaries"].items():
    print(f"[{mode}]")
    print(f"  Success rate   : {s['success_rate_pct']}% ({s['successes']}/{s['total_runs']})")
    print(f"  Avg epochs     : {s['avg_epochs']} +/- {s['avg_epochs_std']}")
    print(f"  Epochs to tau  : {s['avg_epochs_to_tau']} +/- {s['avg_epochs_to_tau_std']}"
          f" (reached: {s['runs_reached_tau']})")
    print(f"  Avg solve time : {s['avg_time_s']} s +/- {s['avg_time_s_std']}")
    print(f"  Final loss     : {s['final_loss_mean']} +/- {s['final_loss_std']}")
    print(f"  Degenerate     : {s['degenerate_pct']}%\\n")

print(payload.get("latex", ""))

# Lưu kết quả lên Google Drive (tuỳ chọn — bỏ comment nếu muốn)
# from google.colab import drive
# drive.mount("/content/drive")
# !cp profiling/ablation_cgl_test*.json /content/drive/MyDrive/
'''

cells = []
for src, kind in ((None, "markdown"), (SETUP, "code"), (INSTALL, "code"),
                  (SMOKE, "code"), (FULL, "code"), (REPORT, "code")):
    if kind == "markdown":
        body = (
            "# Ablation: Random vs Smart Initialization\n"
            "Chạy ablation study cho optimizer hình học khác biệt được trên tập test "
            "[quangne/CGL-Text2Geo](https://huggingface.co/datasets/quangne/CGL-Text2Geo) (381 samples).\n"
            "\n"
            "**Metrics:** Success Rate (%), Avg. Epochs, Epochs-to-τ, Final Loss, "
            "Degenerate Cases (%), Avg. Solve Time.\n"
            "\n"
            "| Khởi tạo | Success Rate (%) | Avg. Epochs | Final Loss | Degenerate (%) |\n"
            "|---|---|---|---|---|\n"
            "| Random | ? | ? | ? | ? |\n"
            "| Smart | ? | ? | ? | ? |\n"
            "\n"
            "**Cách dùng:** chạy lần lượt cell 1 → 5. Cell 4 chạy ~3 giờ; nếu Colab "
            "ngắt kết nối thì chạy lại cell 4 (có resume, không mất kết quả cũ).\n"
        )
    else:
        body = src
    cells.append({
        "cell_type": kind,
        "metadata": {},
        "source": body.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if kind == "code" else {}),
    })

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out = root / "profiling" / "ablation_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes)")
