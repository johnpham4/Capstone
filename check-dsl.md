# Full Sweep DSL + Image Checker (413-Safe)
Ban la Copilot reviewer cho dataset hinh hoc.
Muc tieu: duyet TOAN BO sample trong JSON va kiem tra dong thoi:
1) DSL co dung rule khong
2) Hinh ve co khop de va khop DSL khong

Bat buoc doc rule tai file:
[pipeline/domain/prompt_dsl.py](pipeline/domain/prompt_dsl.py)

Khong bo qua sample nao.

## Nguyen tac thuc thi
- BAT BUOC review thu cong bang Copilot GPT-5.3-Codex tren tung sample (doc de + doc DSL + xem anh).
- KHONG dung code checker/regex checker de ra ket luan PASS/FAIL/REVIEW.
- Chi duoc dung script cho viec doc/ghi file ho tro, KHONG duoc dung script de thay the phan danh gia noi dung.
- Uu tien path trong workspace, khong attach ca folder anh lon de tranh loi 413.
- FULL SWEEP mac dinh: chay het tat ca sample (chi dung khi user noi ro "stop" hoac gioi han max_chunks/max_samples).

## Cau hinh duong dan 
- IMAGE_ROOT=dataset/Minh/images
- OUTPUT_DIR=dataset/Minh
- REVIEW_DIR=dataset/Minh/Review

Quy tac bat buoc:
- Tu muc nay tro xuong, KHONG hard-code path cu the.
- Moi lenh/chi dan phai uu tien dung 3 bien: IMAGE_ROOT, OUTPUT_DIR, REVIEW_DIR.

## Chia sample (bat buoc check truoc khi chay full sweep)
- Muc tieu: chi chia sample khi can thiet, tranh chia lap.
- Thu tu xu ly:
1) Chi duoc coi la "DA CHIA" khi du bo file JSON da chia theo owner (du file va co du lieu hop le).
2) Neu thieu bat ky file nao trong bo JSON da chia (hoac file rong/khong hop le) thi coi la "CHUA CHIA" va PHAI chay script split_full_json.py de chia lai.
3) Sau khi chia xong moi chay full sweep.

Lenh chia (khi can):
- PYTHONPATH=. uv run python split_full_json.py
- Fallback: .\\.venv-win\\Scripts\\python.exe split_full_json.py

Sau khi chia bang split_full_json.py:
- Du lieu tao ra theo owner: <owner_dir>/full.json.
- Khi can chi dinh JSON cho full sweep, uu tien dung file da chia tuong ung owner (vi du: json-files=<owner_dir>/full.json).

## Cach thuc thi (manual)
- Khong chay auto checker de phan loai.
- Duyet theo chunk nho (khuyen nghi 20 sample/chunk) de dam bao chat luong.
- Moi sample BAT BUOC lam 3 buoc:
1) Doc de bai va xac dinh rang buoc hinh hoc.
2) Doc DSL, doi chieu tung rang buoc voi de bai.
3) Xem anh that va doi chieu voi de bai + DSL.

Luu y danh gia:
- Neu de mo ho NHUNG DSL + hinh van khop de bai thi cho PASS.
- Chi cho REVIEW khi de thieu thong tin de dung hinh (vi du noi "duong thang" nhung khong noi di qua dau/quan he nao).
- Neu DSL sai hoac hinh sai (hoac ca 2) thi cho FAIL.

## Input toi thieu
- Duong dan file JSON dataset
- Duong dan thu muc anh goc

Moi sample co the co:
- id hoac image_dir
- instruction
- answer (DSL)
- problem (neu co)

## Quy uoc resolve anh
- Uu tien map theo image_dir neu co.
- Neu image_dir la images/img_<id>.png nhung file thuc te la diagram_<id>.*, fallback theo id voi thu tu:
1) diagram_<id>.png
2) diagram_<id>.jpg / diagram_<id>.jpeg / diagram_<id>.webp
3) img_<id>.png / img_<id>.jpg / img_<id>.jpeg / img_<id>.webp
- Neu khong tim thay anh: gan IMAGE_MISSING va tiep tuc sample tiep theo.

## Bat buoc moi sample
1) Check DSL theo toan bo rule trong prompt_dsl.py
2) Mo anh theo id/image_dir va check hinh co khop DSL
3) Check hinh co khop de (vuong goc, song song, trung diem, duong tron, vi tri diem)
4) Ghi short_reason CU THE theo tung sample, khong dung cau chung chung.

Khong duoc ket luan PASS neu chua check ca DSL va anh.

## Reason codes
- DSL_SYNTAX
- DSL_WRONG_MAPPING
- DSL_MISSING_CONSTRAINT
- DSL_EXTRA_CONSTRAINT
- DSL_POINT_DEFINE
- DSL_ANGLE
- DSL_CIRCLE_CENTER
- DSL_ORDER_DEPENDENCY
- IMAGE_GEOMETRY_MISMATCH
- IMAGE_LABEL_MISMATCH
- IMAGE_LAYOUT_BAD
- IMAGE_MISSING
- AMBIGUOUS_PROBLEM

## Decision rule
- PASS: DSL dung + anh khop DSL + anh khop de
- FAIL: co loi ro rang (DSL hoac hinh)
- REVIEW: chua du chac chan hoac de mo ho
- IMAGE_MISSING: khong tim thay anh

## Output bat buoc
Chi can 4 file split:
- review_pass.json (chi gom sample status PASS)
- review_fail.json (chi gom sample status FAIL)
- review_ambiguous.json (chi gom sample status REVIEW - de mo ho/chua ro)
- review_image_missing.json (chi gom sample status IMAGE_MISSING)

Bat buoc cho file nay:
- Van phai xuat DUNG 4 file split nhu tren.
- Moi item phai co short_reason cu the, mo ta dung loi that su duoc quan sat tu de + DSL + anh.

Bat buoc luu 4 file split trong folder tai REVIEW_DIR:
- Neu chua co folder REVIEW_DIR thi phai tao moi.
- Neu da co folder REVIEW_DIR thi dung lai folder do.

Moi item trong 4 file tren PHAI giu du cac truong goc:
- image_dir
- instruction
- answer
- problem

Sau khi chay xong, chi tra ve DUY NHAT 1 JSON object ngan (khong markdown), gom duong dan 4 file:
{
  "review_pass": "<REVIEW_DIR>/review_pass.json",
  "review_fail": "<REVIEW_DIR>/review_fail.json",
  "review_ambiguous": "<REVIEW_DIR>/review_ambiguous.json",
  "review_image_missing": "<REVIEW_DIR>/review_image_missing.json"
}

Khong in lai toan bo DSL trong chat tru khi can chung minh loi.