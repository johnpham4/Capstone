prompt: str = """
Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL.

═══ CÚ PHÁP DSL ═══

1. HÌNH
   • (triangle (A B C)) / (triangle (A B C) (isosceles A)) / (triangle (A B C) (right B)) / (triangle (A B C) (equilateral))
   • (square (A B C D))

2. ĐIỂM: (define <name> point <construction>)
   • Đặc biệt: (midpoint B C), (centroid A B C), (incenter A B C), (circumcenter A B C), (orthocenter A B C)
   • Hình chiếu: (projection A (segment B C))
   • Phân giác: (bisector B A C) - góc BAC, đỉnh A
   • Giao điểm: (inter-ll C D A B) - giao của CD và AB
   • Tự do: (segment A B), (line A B)
   
   ⚠️ HÌNH CHIẾU: LUÔN vẽ segment từ điểm xuống hình chiếu
   "H là hình chiếu B lên AC" → (define H point (projection B (segment A C))) + (segment B H)
   
   ⚠️ GIAO ĐIỂM: "CD cắt AB tại H" → (define H point (inter-ll C D A B))
   
   ⚠️ CẤM DÙNG on-circle TRONG CONSTRUCTION:
   SAI: (define C point (on-circle C O))
   ĐÚNG: (define C point) + (on-circle C O)
   
   ⚠️ TRUNG ĐIỂM CUNG vs TRUNG ĐIỂM ĐOẠN:
   • "C là trung điểm cung AB" / "C là trung điểm cung nhỏ AB"
     → (define C point) + (on-circle C O)
     → C là điểm TỰ DO trên đường tròn, KHÔNG dùng midpoint!
   • "C là trung điểm đoạn AB" → (define C point (midpoint A B))
   
   ⚠️ CHỨNG MINH vs CHO SẴN:
   • Đề bảo "Chứng minh góc A = 60°" / "Tính góc A" → KHÔNG thêm (angle-measure B A C 60)
   • Đề bảo "Chứng minh OC ⟂ AB" → KHÔNG thêm (perpendicular ...)
   • CHỈ thêm constraint khi đề BÀI CHO SẴN, không phải chứng minh

3. ĐOẠN/ĐƯỜNG
   • (segment A B) - đoạn thẳng
   • (line A B) - đường thẳng

4. ĐƯỜNG TRÒN
   Khai báo:
   • (circle O) hoặc (circle O (radius 0.5))
   • (incircle A B C) / (circumcircle A B C)
   • LUÔN: (define O point) TRƯỚC → (circle O) SAU
   • Scale: **1 cm = 0.1 đơn vị** (5cm → 0.5, 10cm → 1.0)
   
   ⚠️⚠️⚠️ ĐƯỜNG KÍNH vs DÂY - CỰC KỲ QUAN TRỌNG:
   
   **ĐƯỜNG KÍNH** (đi qua tâm O):
   • "đường kính AB" / "đường kính MN" / "có đường kính AB" → BẮT BUỘC 4 thành phần:
     1. (segment A B)
     2. (on-circle A O)
     3. (on-circle B O)
     4. (on-segment O A B) ← TÂM O NẰM GIỮA - TUYỆT ĐỐI KHÔNG THIẾU!
   
   **DÂY THƯỜNG** (không qua tâm):
   • "dây AB" / "dây CD" → CHỈ 3 thành phần:
     1. (segment A B)
     2. (on-circle A O)
     3. (on-circle B O)
   
   VÍ DỤ ĐƯỜNG KÍNH:
   
   a) "Cho đường tròn (O) có đường kính MN"
      (define O point)
      (circle O)
      (define M point)
      (define N point)
      (segment M N)
      (on-circle M O)
      (on-circle N O)
      (on-segment O M N)  ← BẮT BUỘC!
   
   b) "Cho đường tròn (O) có đường kính MN. Một dây AB cắt MN tại H"
      (define O point)
      (circle O)
      (define M point)
      (define N point)
      (segment M N)
      (on-circle M O)
      (on-circle N O)
      (on-segment O M N)  ← Đường kính MN
      (define A point)
      (define B point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)  ← Dây AB - KHÔNG có on-segment
      (define H point (inter-ll A B M N))
   
   c) "đường tròn (O) đường kính AB, C trên đường tròn, H là hình chiếu C lên AB"
      (define O point)
      (circle O)
      (define A point)
      (define B point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (on-segment O A B)  ← Đường kính
      (define C point)
      (on-circle C O)
      (define H point (projection C (segment A B)))
      (segment C H)

5. RÀNG BUỘC
   • (parallel (segment B C) (segment D E))
   • (perpendicular (segment A B) (segment C D))
   • (angle-equal A B C D E F) - ∠ABC = ∠DEF
   • (angle-measure B A C 60) - ∠BAC = 60° (đỉnh A ở GIỮA)
   • (on-segment M C D) - M nằm trên đoạn CD
   • (on-circle A O) - A trên đường tròn tâm O
   • (distance O A 0.5), (equal-distance O M O H)

═══ QUY TẮC ═══

1. THỨ TỰ: Hình → Define points → Segments/Lines → Circles → Constraints

2. GÓC: "góc A = 60°" → (angle-measure B A C 60) [A ở GIỮA]

3. ĐƯỜNG ĐẶC BIỆT:
   • Đường cao AH: (define H point (projection A (segment B C))) + (segment A H)
   • Trung tuyến AM: (define M point (midpoint B C)) + (segment A M)
   • Phân giác AD: (define D point (bisector B A C)) + (segment A D)

═══ VÍ DỤ ═══

1. "Tam giác ABC, M là trung điểm BC"
   (triangle (A B C))
   (define M point (midpoint B C))
   (segment A M)

2. "Tam giác ABC vuông tại B, góc A = 60°"
   (triangle (A B C) (right B))
   (angle-measure B A C 60)

3. "Cho đường tròn (O) với ∠AOB = 120°. Lấy điểm C là trung điểm cung nhỏ AB"
   (define O point)
   (circle O)
   (define A point)
   (define B point)
   (segment O A)
   (segment O B)
   (segment A B)
   (on-circle A O)
   (on-circle B O)
   (angle-measure A O B 120)
   (define C point)
   (on-circle C O)
   (segment A C)
   (segment B C)
   (segment O C)

4. "đường tròn (O) bán kính 6cm, dây AB, góc ở tâm AOB = 120°"
   (define O point)
   (circle O (radius 0.6))
   (define A point)
   (define B point)
   (segment O A)
   (segment O B)
   (segment A B)
   (on-circle A O)
   (on-circle B O)
   (angle-measure A O B 120)

═══ OUTPUT FORMAT ═══
⚠️ TUYỆT ĐỐI QUAN TRỌNG:

1. CHỈ trả về JSON: [{"instruction": "...", "answer": "DSL với \\n"}]

2. Field "instruction" PHẢI là bài toán TIẾNG VIỆT gốc, KHÔNG được thay đổi!
   SAI: "Convert the geometry problem to GMBL"
   ĐÚNG: "Cho đường tròn (O) có đường kính MN..."

3. Field "answer" CHỈ chứa DSL thuần túy:
   • KHÔNG có comment #
   • KHÔNG có giải thích
   • CHỈ có DSL với \\n

4. KHÔNG markdown, KHÔNG giải thích bên ngoài JSON

═══ INPUT ═══
{{ extract }}

"""
