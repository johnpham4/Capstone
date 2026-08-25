import base64, json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
repo_url = "https://github.com/johnpham4/GeoSystem.git"
patch_b64 = base64.b64encode((root / "profiling" / "colab_bundle.patch").read_bytes()).decode("ascii")

def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

md = "# Ablation: Random vs Smart Initialization\n"
md += "So sanh optimizer hinh hoc tren tap test "
md += "[quangne/CGL-Text2Geo](https://huggingface.co/datasets/quangne/CGL-Text2Geo) "
md += "(381 samples), moi DSL toi uu **1 lan/mode**, seed co dinh.\n\n"
md += "**Dau ra:**\n"
md += "- `report.csv` - loss, epochs, runtime, coid degenerate *tham khao*, duong dan anh\n"
md += "- `gallery.html` - anh random | smart dat canh nhau de ban **duyet bang mat**\n"
md += "- Bang LaTeX tong hop o cell cuoi\n\n"
md += "**Cach dung:** chay lan luot cell 1 den 5. Cell 4 chay ~70 phut; "
md += "neu Colab ngat thi chay lai cell 4 (co resume).\n"

s1 = "#@title 1. Clone repo + ap dung patch\n"
s1 += "import base64, pathlib\n\n"
s1 += 'REPO = "%s"\n\n' % repo_url
s1 += "if not pathlib.Path('/content/GeoSystem').exists():\n"
s1 += "    !git clone --depth 1 %s /content/GeoSystem\n\n" % repo_url
s1 += 'PATCH_B64 = "%s"\n' % patch_b64
s1 += "%cd /content/GeoSystem\n"
s1 += 'pathlib.Path("colab_bundle.patch").write_bytes(base64.b64decode(PATCH_B64))\n'
s1 += '!git apply colab_bundle.patch && echo "PATCH OK"\n'

s2 = "#@title 2. Cai dependencies\n"
s2 += "!pip -q install loguru datasets matplotlib\n"

s3 = "#@title 3. Smoke test (~4 phut) - co render anh xem thu gallery\n"
s3 += "!python -u profiling/ablation_initializer.py --repo quangne/CGL-Text2Geo --split test --field answer --max-samples 6 --seeds 1 --epochs 1000 --workers 2 --render-dir profiling/render --output profiling/colab_smoke.json\n\n"
s3 += "from IPython.display import HTML\n"
s3 += 'HTML(filename="profiling/render/gallery.html")\n'

s4 = "#@title 4. CHAY FULL - moi DSL toi uu 1 lan/mode, kem anh (~70 phut Colab free)\n"
s4 += "#@markdown Co --resume: neu Colab ngat, chay lai cell nay - tiep tuc tu cho dung.\n"
s4 += "!python -u profiling/ablation_initializer.py --repo quangne/CGL-Text2Geo --split test --field answer --seeds 1 --epochs 1000 --workers 2 --render-dir profiling/render --output profiling/ablation_cgl_test.json\n"

s5 = "#@title 5. Ket qua tong hop + bang LaTeX + gallery\n"
s5 += "import json\n"
s5 += "from IPython.display import HTML\n\n"
s5 += "with open('profiling/ablation_cgl_test.json', encoding='utf-8') as f:\n"
s5 += "    payload = json.load(f)\n\n"
s5 += 'print("Elapsed:", payload["elapsed_seconds"], "s")\n'
s5 += "for mode, s in payload['summaries'].items():\n"
s5 += "    print()\n"
s5 += "    print(f'[{mode}]')\n"
s5 += "    print(f'  Success rate   : {s[\"success_rate_pct\"]}% ({s[\"successes\"]}/{s[\"total_runs\"]})')\n"
s5 += "    print(f'  Avg epochs     : {s[\"avg_epochs\"]} +/- {s[\"avg_epochs_std\"]}')\n"
s5 += "    print(f'  Epochs to tau  : {s[\"avg_epochs_to_tau\"]} +/- {s[\"avg_epochs_to_tau_std\"]} (reached: {s[\"runs_reached_tau\"]})')\n"
s5 += "    print(f'  Avg solve time : {s[\"avg_time_s\"]} s +/- {s[\"avg_time_s_std\"]}')\n"
s5 += "    print(f'  Final loss     : {s[\"final_loss_mean\"]} +/- {s[\"final_loss_std\"]}')\n"
s5 += "    print(f'  Degenerate     : {s[\"degenerate_pct\"]}%')\n\n"
s5 += "print()\n"
s5 += "print(payload.get('latex', ''))\n\n"
s5 += "HTML(filename='profiling/render/gallery.html')\n"
s5 += "# from google.colab import drive\n"
s5 += "# drive.mount('/content/drive')\n"
s5 += "# !cp profiling/ablation_cgl_test*.json /content/drive/MyDrive/\n"
s5 += "# !zip -r /content/drive/MyDrive/render.zip profiling/render/\n"

cells = [cell("markdown", md), cell("code", s1), cell("code", s2),
         cell("code", s3), cell("code", s4), cell("code", s5)]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
out = root / "profiling" / "ablation_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes)")
