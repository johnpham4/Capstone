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

═══ THỨ TỰ KHAI BÁO ═══
1. Triangle
2. Define points (midpoint, centroid, incenter, etc.)
3. Segments/Lines
4. Circles
5. Constraints (parallel, perpendicular)

═══ VÍ DỤ ═══

VD1: "Tam giác ABC, M là trung điểm BC"
→ (triangle (A B C))
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

═══ OUTPUT FORMAT ═══
JSON array only:
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