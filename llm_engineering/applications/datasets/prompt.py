prompt: str = """
Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL (S-expression syntax).

═══ CÚ PHÁP DSL ═══

1. TAM GIÁC: (triangle (A B C) [type])
   Types: (isosceles A) | (right B) | (right_isosceles B) | (equilateral)
   - Tam giác thường: (triangle (A B C))
   - Cân tại A: (triangle (A B C) (isosceles A))
   - Vuông tại B: (triangle (A B C) (right B))
   - Vuông cân tại B: (triangle (A B C) (right_isosceles B))
   - Đều: (triangle (A B C) (equilateral))

2. ĐỊNH NGHĨA ĐIỂM: (define <name> point <construction>)
   - (midpoint B C) - trung điểm
   - (centroid A B C) - trọng tâm
   - (orthocenter A B C) - trực tâm
   - (incenter A B C) - tâm nội tiếp
   - (circumcenter A B C) - tâm ngoại tiếp
   - (projection A (segment B C)) - hình chiếu/chân đường cao
   - (segment A B) - điểm trên đoạn thẳng
   - (line A B) - điểm trên đường thẳng

3. ĐOẠN THẲNG: (segment A B)
   - Dùng khi: "đoạn thẳng", "vẽ", "nối", "cạnh"

4. ĐƯỜNG THẲNG: (line A B)
   - Dùng khi: "đường thẳng" (kéo dài vô hạn)

5. ĐƯỜNG TRÒN: (circle <center> <type>)
   - (incircle A B C) - đường tròn nội tiếp
   - (circumcircle A B C) - đường tròn ngoại tiếp
   ⚠️ PHẢI khai báo CẢ tâm (define point) VÀ đường tròn (circle)

6. RÀNG BUỘC:
   - (parallel (segment B C) (segment D E)) - song song
   - (perpendicular (segment A B) (segment C D)) - vuông góc
   ⚠️ PHẢI khai báo segment/line TRƯỚC khi dùng ràng buộc

═══ TỪ KHÓA TIẾNG VIỆT ═══
- "trung điểm" → midpoint
- "trọng tâm" → centroid
- "trực tâm" → orthocenter
- "tâm nội tiếp" / "đường tròn nội tiếp" → incenter / incircle
- "tâm ngoại tiếp" / "đường tròn ngoại tiếp" → circumcenter / circumcircle
- "đường cao" / "chân đường cao" / "hình chiếu" → projection
- "nằm trên đoạn" / "thuộc đoạn" → (segment A B)
- "cân tại" → isosceles
- "vuông tại" → right
- "đều" → equilateral
- "song song" / "//" → parallel
- "vuông góc" / "⊥" → perpendicular

═══ TRƯỜNG HỢP ĐẶC BIỆT ═══

1. TRUNG TUYẾN: "Trung tuyến AM" = Trung điểm M + vẽ đoạn AM
   → (define M point (midpoint B C))
      (segment A M)

2. ĐƯỜNG CAO: "Đường cao AH" = Hình chiếu H + vẽ đoạn AH
   → (define H point (projection A (segment B C)))
      (segment A H)

3. ĐƯỜNG THẲNG VUÔNG GÓC: "Đường thẳng qua C vuông góc AB"
   → (define H point (projection C (segment A B)))
      (line C H)

4. ĐƯỜNG TRUNG BÌNH: "Đường trung bình DE"
   → (define D point (midpoint A B))
      (define E point (midpoint A C))
      (segment D E)

5. ĐƯỜNG TRÒN: LUÔN khai báo tâm TRƯỚC, rồi đường tròn SAU
   - "Đường tròn nội tiếp tâm I"
     → (define I point (incenter A B C))
        (circle I (incircle A B C))
   - "Đường tròn ngoại tiếp tâm O"
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
   (define O point (midpoint A C))

b) Trung điểm cạnh:
   "M là trung điểm AB" →
   (define M point (midpoint A B))

c) Đường chéo:
   "Đường chéo AC" → (segment A C)
   "Hai đường chéo AC và BD" → (segment A C) (segment B D)

d) Hình vuông nội tiếp đường tròn:
   "Hình vuông ABCD nội tiếp đường tròn tâm O" →
   (square (A B C D))
   (define O point (midpoint A C))
   (circle O (circumcircle A B C D))

e) Đường tròn nội tiếp hình vuông:
   "Hình vuông ABCD có đường tròn nội tiếp tâm I" →
   (square (A B C D))
   (define I point (midpoint A C))
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
(define O point (midpoint A C))
(segment A C)
(segment B D)

"Hình vuông ABCD có AC vuông góc BD" → (square (A B C D))
(Không cần assert vì đường chéo hình vuông luôn vuông góc)

"Hình vuông ABCD có hai đường chéo bằng nhau" → (square (A B C D))
(Không cần assert vì đường chéo hình vuông luôn bằng nhau)

"Hình vuông ABCD, O là trung điểm của AC và BD" →
(square (A B C D))
(define O point (midpoint A C))

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

VD2: "Tam giác ABC vuông tại B, đường tròn nội tiếp O"
→ (triangle (A B C) (right B))
   (define O point (incenter A B C))
   (circle O (incircle A B C))

VD3: "Tam giác ABC cân tại A, đường cao AH"
→ (triangle (A B C) (isosceles A))
   (define H point (projection A (segment B C)))
   (segment A H)

VD4: "Tam giác ABC, D trên AB, E trên AC, BC // DE"
→ (triangle (A B C))
   (define D point (segment A B))
   (define E point (segment A C))
   (segment B C)
   (segment D E)
   (parallel (segment B C) (segment D E))

VD5: "Tam giác ABC, đường thẳng qua C vuông góc AB"
→ (triangle (A B C))
   (define H point (projection C (segment A B)))
   (line C H)

VD6: "Tam giác ABC, trọng tâm G, tâm ngoại tiếp O"
→ (triangle (A B C))
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
(define O point (midpoint A C))
(segment A C)
(segment B D)

Input: "Hình vuông ABCD có AC vuông góc BD"
Output: (square (A B C D))

Ví dụ 11: Hình vuông với tâm
Input: "Hình vuông ABCD, O là trung điểm của AC và BD"
Output:
(square (A B C D))
(define O point (midpoint A C))

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
(define O point (midpoint A C))
(circle O (circumcircle A B C D))

Ví dụ 16: Hình vuông ngoại tiếp đường tròn
Input: "Hình vuông ABCD có đường tròn nội tiếp"
Output:
(square (A B C D))
(define I point (midpoint A C))
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
- "tâm hình vuông" / "giao điểm đường chéo" / "trung điểm đường chéo" → midpoint
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
    "instruction": "Nguyên văn tiếng Việt",
    "answer": "DSL code với \\n ngăn cách"
  }
]

Không markdown, không giải thích, chỉ JSON thuần.

═══ INPUT ═══
{{ extract }}
"""