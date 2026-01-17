prompt: str = """
Convert Vietnamese geometry problems to Geometry DSL (Domain Specific Language).

Bạn đang dịch ĐỀ BÀI / CAPTION HÌNH HỌC TIẾNG VIỆT sang Geometry DSL.
Hãy đọc kỹ từng câu tiếng Việt, xác định đúng đối tượng hình học, quan hệ và ràng buộc,
sau đó chuyển đổi chính xác sang cú pháp DSL bên dưới.

=========================================
=== CÚ PHÁP GEOMETRY DSL CƠ BẢN ===
=========================================

DSL này được thiết kế để mô tả hình học một cách chính xác và tối giản.

══════════════════════════
1. TAM GIÁC (TRIANGLE)
══════════════════════════

Cú pháp: (triangle (A B C) [loại_tam_giác])

Các loại tam giác:
- Tam giác thường: (triangle (A B C))
- Tam giác cân tại A: (triangle (A B C) (isosceles A))
- Tam giác vuông tại B: (triangle (A B C) (right B))
- Tam giác vuông cân tại B: (triangle (A B C) (right_isosceles B))
- Tam giác đều: (triangle (A B C) (equilateral))

Ví dụ:
"Tam giác ABC" → (triangle (A B C))
"Tam giác ABC cân tại A" → (triangle (A B C) (isosceles A))
"Tam giác ABC vuông tại B" → (triangle (A B C) (right B))

══════════════════════════
2. ĐỊNH NGHĨA ĐIỂM (DEFINE POINT)
══════════════════════════

Cú pháp: (define <tên_điểm> point <cách_xác_định>)

Các cách xác định điểm:

a) Trung điểm:
   "M là trung điểm BC" → (define M point (midpoint B C))

b) Trọng tâm:
   "G là trọng tâm tam giác ABC" → (define G point (centroid A B C))

c) Trực tâm:
   "H là trực tâm tam giác ABC" → (define H point (orthocenter A B C))

d) Tâm đường tròn nội tiếp:
   "I là tâm đường tròn nội tiếp tam giác ABC" → (define I point (incenter A B C))

e) Tâm đường tròn ngoại tiếp:
   "O là tâm đường tròn ngoại tiếp tam giác ABC" → (define O point (circumcenter A B C))

f) Hình chiếu / Chân đường cao:
   "H là chân đường cao từ A xuống BC" → (define H point (projection A (segment B C)))

g) Điểm trên đoạn thẳng:
   "D nằm trên đoạn AB" → (define D point (segment A B))

h) Giao điểm hai đoạn thẳng:
   "O là giao điểm AC và BD" → (define O point (intersection (segment A C) (segment B D)))

══════════════════════════
3. ĐOẠN THẲNG (SEGMENT)
══════════════════════════

Cú pháp: (segment A B)

Ví dụ:
"Vẽ đoạn AM" → (segment A M)
"Nối A với M" → (segment A M)
"Đường chéo AC" → (segment A C)

══════════════════════════
4. ĐƯỜNG TRÒN (CIRCLE)
══════════════════════════

Cú pháp: (circle <tâm> <cách_xác_định>)

a) Đường tròn nội tiếp:
   "Đường tròn nội tiếp tam giác ABC tâm I"
   → (define I point (incenter A B C))
      (circle I (incircle A B C))

b) Đường tròn ngoại tiếp:
   "Đường tròn ngoại tiếp tam giác ABC tâm O"
   → (define O point (circumcenter A B C))
      (circle O (circumcircle A B C))

⚠️ QUAN TRỌNG:
- Khi có đường tròn tiếp xúc tam giác, PHẢI khai báo CẢ tâm (define point) VÀ đường tròn (circle)
- TUYỆT ĐỐI KHÔNG chỉ khai báo tâm mà bỏ qua đường tròn

══════════════════════════
5. HÌNH VUÔNG (SQUARE)
══════════════════════════

Cú pháp: (square (A B C D))

⚠️ QUY TẮC QUAN TRỌNG:
- Luôn khai báo hình vuông đơn giản: (square (A B C D))
- KHÔNG cần khai báo thêm các thuộc tính tự nhiên của hình vuông như:
  • AB = BC = CD = DA (4 cạnh bằng nhau)
  • Góc ABC = 90° (các góc vuông)
  • AB ⟂ BC (cạnh vuông góc)
  • AB ∥ CD, BC ∥ AD (cạnh đối song song)
  • AC = BD (đường chéo bằng nhau)
  • AC ⟂ BD (đường chéo vuông góc)
- CHỈ khai báo các điểm đặc biệt hoặc đường tròn phụ thuộc nếu đề bài YÊU CẦU

Điểm đặc biệt của hình vuông:

a) Tâm hình vuông (giao điểm 2 đường chéo):
   "Hình vuông ABCD, O là tâm" → 
   (square (A B C D))
   (define O point (intersection (segment A C) (segment B D)))

b) Trung điểm cạnh:
   "M là trung điểm AB" →
   (define M point (midpoint A B))

c) Đường chéo:
   "Đường chéo AC" → (segment A C)
   "Hai đường chéo AC và BD" → (segment A C) (segment B D)

d) Hình vuông nội tiếp đường tròn:
   "Hình vuông ABCD nội tiếp đường tròn tâm O" →
   (square (A B C D))
   (define O point (intersection (segment A C) (segment B D)))
   (circle O (circumcircle A B C D))

e) Đường tròn nội tiếp hình vuông:
   "Hình vuông ABCD có đường tròn nội tiếp tâm I" →
   (square (A B C D))
   (define I point (intersection (segment A C) (segment B D)))
   (circle I (incircle A B C D))

f) Hình vuông bị chia cắt:
   "Hình vuông được chia bởi đường chéo AC" →
   (square (A B C D))
   (segment A C)

Ví dụ:
"Hình vuông ABCD" → (square (A B C D))

"Hình vuông ABCD có cạnh AB = 4" → (square (A B C D))
(Không cần assert AB = 4 vì chỉ mô tả hình dạng)

"Hình vuông ABCD có góc ABC = 90 độ" → (square (A B C D))
(Không cần assert góc vì hình vuông tự nhiên có 4 góc vuông)

"Hình vuông ABCD có AB vuông góc BC" → (square (A B C D))
(Không cần assert vuông góc vì là thuộc tính tự nhiên)

"Hình vuông ABCD có AB song song CD, BC song song AD" → (square (A B C D))
(Không cần assert song song vì là thuộc tính tự nhiên)

"Hình vuông ABCD có đường chéo AC" →
(square (A B C D))
(segment A C)

"Hình vuông ABCD có hai đường chéo AC và BD" →
(square (A B C D))
(segment A C)
(segment B D)

"Hình vuông ABCD, hai đường chéo AC và BD cắt nhau tại O" →
(square (A B C D))
(define O point (intersection (segment A C) (segment B D)))
(segment A C)
(segment B D)

"Hình vuông ABCD có AC vuông góc BD" → (square (A B C D))
(Không cần assert vì đường chéo hình vuông luôn vuông góc)

"Hình vuông ABCD có hai đường chéo bằng nhau" → (square (A B C D))
(Không cần assert vì đường chéo hình vuông luôn bằng nhau)

"Hình vuông ABCD, O là trung điểm của AC và BD" →
(square (A B C D))
(define O point (intersection (segment A C) (segment B D)))

"Hình vuông ABCD, M là trung điểm của AB" →
(square (A B C D))
(define M point (midpoint A B))

"Hình vuông ABCD, M và N lần lượt là trung điểm của AB và BC" →
(square (A B C D))
(define M point (midpoint A B))
(define N point (midpoint B C))

"Hình vuông ABCD, xét tam giác ABC" →
(square (A B C D))
(triangle (A B C))

"Hình vuông ABCD được chia bởi đường chéo AC" →
(square (A B C D))
(segment A C)

"Hình vuông ABCD nội tiếp đường tròn" →
(square (A B C D))
(define O point (intersection (segment A C) (segment B D)))
(circle O (circumcircle A B C D))

"Hình vuông ABCD có đường tròn nội tiếp" →
(square (A B C D))
(define I point (intersection (segment A C) (segment B D)))
(circle I (incircle A B C D))

════════════════════════════════
=== QUY TẮC QUAN TRỌNG ===
════════════════════════════════

1. Thứ tự khai báo:
   - Khai báo tam giác/hình vuông TRƯỚC TIÊN (luôn là dòng đầu tiên)
   - Định nghĩa điểm phụ thuộc SAU
   - Vẽ đoạn thẳng / đường tròn CUỐI CÙNG
   
   ⚠️ QUAN TRỌNG: LUÔN khai báo hình (triangle/square) trước, kể cả khi có tâm đường tròn

2. Tên điểm:
   - Sử dụng chữ cái in hoa: A, B, C, D, E, F, G, H, I, M, N, O...
   - Giữ nguyên tên từ đề bài tiếng Việt

3. Loại tam giác:
   - (right B) nghĩa là góc B = 90°
   - (isosceles A) nghĩa là AB = AC (cân tại A)
   - KHÔNG cần assert thêm các thuộc tính đã có trong khai báo tam giác

4. Hình vuông:
   - Chỉ khai báo: (square (A B C D))
   - KHÔNG assert các thuộc tính tự nhiên (cạnh bằng nhau, góc vuông, song song, vuông góc)
   - CHỈ khai báo điểm đặc biệt, đoạn thẳng, đường tròn khi đề bài YÊU CẦU
   - Đường chéo AC, BD cần khai báo bằng (segment A C), (segment B D)

5. Đường tròn:
   - LUÔN phải có (define ... point ...) cho tâm
   - SAU ĐÓ mới có (circle tâm ...)
   - Không được bỏ qua bất kỳ cái nào

6. Đơn giản hóa:
   - Nếu đề bài chỉ nói "Hình vuông ABCD" mà không đề cập gì thêm → CHỈ cần (square (A B C D))
   - Nếu đề bài nói "có góc vuông", "cạnh bằng nhau", "song song", "vuông góc" → Vẫn CHỈ cần (square (A B C D))
   - CHỈ thêm khai báo khi đề bài yêu cầu điểm đặc biệt, đoạn thẳng, đường tròn cụ thể

════════════════════════════════
=== VÍ DỤ HOÀN CHỈNH ===
════════════════════════════════

Ví dụ 1: Đơn giản
Input: "Tam giác ABC"
Output: (triangle (A B C))

Ví dụ 2: Tam giác với trung điểm
Input: "Tam giác ABC, M là trung điểm BC"
Output:
(triangle (A B C))
(define M point (midpoint B C))
(segment A M)

Ví dụ 3: Tam giác cân với đường cao
Input: "Tam giác ABC cân tại A, có đường cao AH"
Output:
(triangle (A B C) (isosceles A))
(define H point (projection A (segment B C)))
(segment A H)

Ví dụ 4: Tam giác với đường tròn nội tiếp
Input: "Tam giác ABC có đường tròn nội tiếp tâm I"
Output:
(triangle (A B C))
(define I point (incenter A B C))
(circle I (incircle A B C))

Ví dụ 5: Tam giác vuông với nhiều điểm đặc biệt
Input: "Tam giác ABC vuông tại B, D là trung điểm AB, E là trung điểm AC, F là trung điểm BC"
Output:
(triangle (A B C) (right B))
(define D point (midpoint A B))
(define E point (midpoint A C))
(define F point (midpoint B C))
(segment D E)
(segment E F)
(segment B C)

Ví dụ 6: Tam giác với trực tâm, trọng tâm, tâm ngoại tiếp
Input: "Tam giác ABC vuông tại B, có trực tâm H, trọng tâm G, tâm đường tròn ngoại tiếp O"
Output:
(triangle (A B C) (right B))
(define H point (orthocenter A B C))
(define G point (centroid A B C))
(define O point (circumcenter A B C))
(circle O (circumcircle A B C))

Ví dụ 7: Tam giác cân với trung điểm và đường tròn nội tiếp
Input: "Tam giác ABC cân tại A, M là trung điểm BC, I là tâm đường tròn nội tiếp"
Output:
(triangle (A B C) (isosceles A))
(define M point (midpoint B C))
(define I point (incenter A B C))
(circle I (incircle A B C))
(segment A M)

Ví dụ 8: Hình vuông đơn giản
Input: "Hình vuông ABCD"
Output: (square (A B C D))

Ví dụ 9: Hình vuông với thuộc tính tự nhiên (KHÔNG cần assert)
Input: "Hình vuông ABCD có cạnh AB = 4"
Output: (square (A B C D))

Input: "Hình vuông ABCD có góc ABC = 90 độ"
Output: (square (A B C D))

Input: "Hình vuông ABCD có AB vuông góc BC"
Output: (square (A B C D))

Input: "Hình vuông ABCD có AB song song CD, BC song song AD"
Output: (square (A B C D))

Input: "Hình vuông ABCD có hai đường chéo bằng nhau"
Output: (square (A B C D))

Ví dụ 10: Hình vuông với đường chéo
Input: "Hình vuông ABCD có đường chéo AC"
Output:
(square (A B C D))
(segment A C)

Input: "Hình vuông ABCD có hai đường chéo AC và BD"
Output:
(square (A B C D))
(segment A C)
(segment B D)

Input: "Hình vuông ABCD, hai đường chéo AC và BD cắt nhau tại O"
Output:
(square (A B C D))
(define O point (intersection (segment A C) (segment B D)))
(segment A C)
(segment B D)

Input: "Hình vuông ABCD có AC vuông góc BD"
Output: (square (A B C D))

Ví dụ 11: Hình vuông với tâm
Input: "Hình vuông ABCD, O là trung điểm của AC và BD"
Output:
(square (A B C D))
(define O point (intersection (segment A C) (segment B D)))

Ví dụ 12: Hình vuông với trung điểm cạnh
Input: "Hình vuông ABCD, M là trung điểm của AB"
Output:
(square (A B C D))
(define M point (midpoint A B))

Input: "Hình vuông ABCD, M và N lần lượt là trung điểm của AB và BC"
Output:
(square (A B C D))
(define M point (midpoint A B))
(define N point (midpoint B C))

Ví dụ 13: Hình vuông với tam giác
Input: "Hình vuông ABCD, xét tam giác ABC"
Output:
(square (A B C D))
(triangle (A B C))

Ví dụ 14: Hình vuông bị chia cắt
Input: "Hình vuông ABCD được chia bởi đường chéo AC"
Output:
(square (A B C D))
(segment A C)

Ví dụ 15: Hình vuông nội tiếp đường tròn
Input: "Hình vuông ABCD nội tiếp đường tròn"
Output:
(square (A B C D))
(define O point (intersection (segment A C) (segment B D)))
(circle O (circumcircle A B C D))

Ví dụ 16: Hình vuông ngoại tiếp đường tròn
Input: "Hình vuông ABCD có đường tròn nội tiếp"
Output:
(square (A B C D))
(define I point (intersection (segment A C) (segment B D)))
(circle I (incircle A B C D))

════════════════════════════════
=== LƯU Ý DỊCH TIẾNG VIỆT ===
════════════════════════════════

Từ khóa thường gặp:
- "trung điểm" → midpoint
- "trọng tâm" → centroid
- "trực tâm" → orthocenter
- "tâm đường tròn nội tiếp" → incenter
- "tâm đường tròn ngoại tiếp" → circumcenter
- "đường cao" / "chân đường cao" → projection
- "nằm trên đoạn" → segment
- "cân tại" → isosceles
- "vuông tại" → right
- "tam giác đều" → equilateral
- "hình vuông" → square
- "tâm hình vuông" / "giao điểm đường chéo" → intersection
- "đường chéo" → segment (AC hoặc BD)
- "cắt nhau tại" / "giao điểm" → intersection
- "nội tiếp đường tròn" → inscribed (circumcircle)
- "ngoại tiếp đường tròn" / "đường tròn nội tiếp" → circumscribed (incircle)

════════════════════════════════
=== ĐỊNH DẠNG ĐẦU RA ===
════════════════════════════════

CỰC KỲ QUAN TRỌNG 

CHỈ trả về JSON array với ĐÚNG format sau:

[
  {
    "instruction": "Nguyên văn đề bài tiếng Việt",
    "answer": "Mã DSL với \\n ngăn cách các dòng lệnh"
  }
]

QUY TẮC BẮT BUỘC:
- CHỈ có 2 field: "instruction" và "answer"
- TUYỆT ĐỐI KHÔNG thêm field "id", "image_dir" hay bất kỳ field nào khác
- "answer" là STRING hoàn chỉnh, các lệnh ngăn cách bằng \\n
- KHÔNG cắt cụt answer, PHẢI viết đầy đủ đến hết
- KHÔNG thêm markdown code blocks (```), KHÔNG giải thích
- KHÔNG có ký tự thừa, khoảng trắng thừa

NẾU KHÔNG THỂ GENERATE ĐẦY ĐỦ, HÃY BỎ QUA SAMPLE ĐÓ

SAI:
[
  {
    "id": "abc123",
    "image_dir": "images/img_210001.png",
    "instruction": "Hình vuông ABCD có cạnh AB = 4",
    "answer": "(square (A B"
  }
]

ĐÚNG:
[
  {
    "instruction": "Hình vuông ABCD có cạnh AB = 4",
    "answer": "(square (A B C D))"
  },
  {
    "instruction": "Tam giác ABC cân tại A, có đường cao AH",
    "answer": "(triangle (A B C) (isosceles A))\\n(define H point (projection A (segment B C)))\\n(segment A H)"
  }
]
"""