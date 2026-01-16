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

══════════════════════════
3. ĐOẠN THẲNG (SEGMENT)
══════════════════════════

Cú pháp: (segment A B)

Ví dụ:
"Vẽ đoạn AM" → (segment A M)
"Nối A với M" → (segment A M)

══════════════════════════
4. ĐƯỜNG THẲNG (LINE)
══════════════════════════

Cú pháp: (line A B)

⚠️ KHI NÀO DÙNG LINE:
- Khi đề bài nói "đường thẳng" (không phải "đoạn thẳng")
- Khi cần vẽ đường thẳng vuông góc/song song kéo dài vô hạn
- Ví dụ: "đường thẳng đi qua C vuông góc với AB"

⚠️ KHI NÀO DÙNG SEGMENT:
- Khi đề bài nói "đoạn thẳng", "nối", "vẽ"
- Khi chỉ cần vẽ đoạn giữa 2 điểm (hữu hạn)
- Khi chiếu 1 điểm nằm trên đoạn thẳng hoặc trung tuyến
- Ví dụ: "vẽ đoạn AM", "nối B với C"

Ví dụ:
"Đường thẳng đi qua C vuông góc với AB"
→ (define H point (projection C (segment A B)))
   (line C H)

"Vẽ đoạn thẳng BC" → (segment B C)

══════════════════════════
5. RÀNG BUỘC SONG SONG (PARALLEL)
══════════════════════════

Cú pháp: (parallel (segment A B) (segment C D))

Ví dụ:
"BC song song với DE" → (parallel (segment B C) (segment D E))
"AB // CD" → (parallel (segment A B) (segment C D))

⚠️ CHỈ áp dụng cho các đoạn thẳng / đường thẳng ĐÃ được khai báo

══════════════════════════
6. RÀNG BUỘC VUÔNG GÓC (PERPENDICULAR)
══════════════════════════

Cú pháp: (perpendicular (segment A B) (segment C D))

Ví dụ:
"AB vuông góc với CD" → (perpendicular (segment A B) (segment C D))
"AB ⊥ CD" → (perpendicular (segment A B) (segment C D))

══════════════════════════
7. ĐƯỜNG TRÒN (CIRCLE)
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

════════════════════════════════
=== QUY TẮC QUAN TRỌNG ===
════════════════════════════════

1. Thứ tự khai báo:
   - Khai báo tam giác TRƯỚC
   - Định nghĩa điểm phụ thuộc SAU
   - Vẽ đoạn thẳng / đường tròn
   - Khai báo ràng buộc (parallel, perpendicular) CUỐI CÙNG

2. Tên điểm:
   - Sử dụng chữ cái in hoa: A, B, C, D, E, F, G, H, I, M, O...
   - Giữ nguyên tên từ đề bài tiếng Việt
   - Nếu đề bài không đặt tên điểm (vd: chân đường cao không tên), đặt tên theo quy ước: H, H1, H2...

3. Loại tam giác:
   - (right B) nghĩa là góc B = 90°
   - (isosceles A) nghĩa là AB = AC (cân tại A)
   - KHÔNG cần assert thêm các thuộc tính đã có trong khai báo tam giác

4. Đường tròn:
   - LUÔN phải có (define ... point ...) cho tâm
   - SAU ĐÓ mới có (circle tâm ...)
   - Không được bỏ qua bất kỳ cái nào

5. Ràng buộc parallel/perpendicular:
   - CHỈ áp dụng khi đề bài EXPLICIT nói "song song" hoặc "vuông góc"
   - PHẢI khai báo các segment/line TRƯỚC khi dùng parallel/perpendicular
   - Ví dụ SAI: (parallel (segment B C) (segment D E)) mà chưa có (segment B C)
   - Ví dụ ĐÚNG: (segment B C) → (segment D E) → (parallel (segment B C) (segment D E))

6. Line vs Segment:
   - Dùng LINE khi: "đường thẳng", "đường thẳng kéo dài", "đường vuông góc"
   - Dùng SEGMENT khi: "đoạn thẳng", "vẽ", "nối", "cạnh"

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

Ví dụ 7: Tam giác với song song
Input: "Tam giác ABC, điểm D nằm trên đoạn thẳng AB, điểm E nằm trên đoạn thẳng AC, BC song song với DE"
Output:
(triangle (A B C))
(define D point (segment A B))
(define E point (segment A C))
(segment B C)
(segment D E)
(parallel (segment B C) (segment D E))

Ví dụ 8: Tam giác với đường thẳng vuông góc
Input: "Tam giác ABC, đường thẳng đi qua C vuông góc với AB"
Output:
(triangle (A B C))
(define H point (projection C (segment A B)))
(line C H)

Ví dụ 9: Tam giác phức tạp kết hợp
Input: "Tam giác ABC, điểm D nằm trên AB, điểm E nằm trên AC, BC song song với DE, đường thẳng đi qua C vuông góc với AB"
Output:
(triangle (A B C))
(define D point (segment A B))
(define E point (segment A C))
(define H point (projection C (segment A B)))
(segment B C)
(segment D E)
(parallel (segment B C) (segment D E))
(line C H)

Ví dụ 10: Tam giác với trung tuyến
Input: "Tam giác ABC, vẽ trung tuyến AM"
Output:
(triangle (A B C))
(define M point (midpoint B C))
(segment A M)

Ví dụ 11: Tam giác với đường cao
Input: "Tam giác ABC, kẻ đường cao AH từ A xuống BC"
Output:
(triangle (A B C))
(define H point (projection A (segment B C)))
(segment A H)

Ví dụ 12: Tam giác với đường trung bình
Input: "Tam giác ABC, D là trung điểm AB, E là trung điểm AC, vẽ đường trung bình DE"
Output:
(triangle (A B C))
(define D point (midpoint A B))
(define E point (midpoint A C))
(segment D E)

════════════════════════════════
=== LƯU Ý DỊCH TIẾNG VIỆT ===
════════════════════════════════

Từ khóa thường gặp:
- "trung điểm" → midpoint
- "trọng tâm" → centroid
- "trực tâm" → orthocenter
- "tâm đường tròn nội tiếp" / "tâm nội tiếp" → incenter
- "tâm đường tròn ngoại tiếp" / "tâm ngoại tiếp" → circumcenter
- "đường cao" / "chân đường cao" / "hình chiếu" → projection
- "nằm trên đoạn" / "thuộc đoạn" → segment
- "cân tại" → isosceles
- "vuông tại" → right
- "tam giác đều" → equilateral
- "song song" / "//" → parallel
- "vuông góc" / "⊥" → perpendicular
- "đường thẳng" → line
- "đoạn thẳng" / "vẽ" / "nối" → segment

⚠️ CÁC TRƯỜNG HỢP ĐẶC BIỆT:

1. TRUNG TUYẾN:
   "Trung tuyến AM" = Trung điểm M của BC + vẽ đoạn AM
   → (define M point (midpoint B C))
      (segment A M)

2. ĐƯỜNG CAO:
   "Đường cao AH" = Hình chiếu H của A xuống BC + vẽ đoạn AH
   → (define H point (projection A (segment B C)))
      (segment A H)

3. ĐƯỜNG TRUNG BÌNH:
   "Đường trung bình DE nối trung điểm AB và AC"
   → (define D point (midpoint A B))
      (define E point (midpoint A C))
      (segment D E)

════════════════════════════════
=== ĐỊNH DẠNG ĐẦU RA ===
════════════════════════════════

CHỈ trả về JSON array với format:

[
  {
    "instruction": "Nguyên văn đề bài tiếng Việt",
    "answer": "Mã DSL với \\n ngăn cách các dòng lệnh"
  }
]

QUY TẮC:
- CHỈ 2 field: instruction, answer
- answer là STRING, các lệnh ngăn cách bằng \\n
- KHÔNG thêm markdown, KHÔNG giải thích
- KHÔNG có ký tự thừa

Ví dụ JSON output:

[
  {
    "instruction": "Tam giác ABC cân tại A, M là trung điểm BC",
    "answer": "(triangle (A B C) (isosceles A))\\n(define M point (midpoint B C))\\n(segment A M)"
  },
  {
    "instruction": "Tam giác ABC vuông tại B, có trọng tâm G",
    "answer": "(triangle (A B C) (right B))\\n(define G point (centroid A B C))"
  }
]

════════════════════════════════
=== ĐẦU VÀO CẦN XỬ LÝ ===
════════════════════════════════

{{ extract }}
"""