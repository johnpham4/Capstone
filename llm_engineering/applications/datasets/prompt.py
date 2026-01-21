prompt: str = """
Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL (S-expression syntax).

═══ CÚ PHÁP DSL ═══

1. HÌNH (SHAPES)
   • Tam giác: (triangle (A B C) [type])
     - Thường: (triangle (A B C))
     - Cân: (triangle (A B C) (isosceles A))
     - Vuông: (triangle (A B C) (right B))
     - Vuông cân: (triangle (A B C) (right_isosceles B))
     - Đều: (triangle (A B C) (equilateral))
     - Tù: (triangle (A B C) (obtuse B))  # B là đỉnh góc tù

   • Hình vuông: (square (A B C D))
   CHỈ khai báo (square ...), KHÔNG assert thuộc tính tự nhiên (cạnh bằng nhau, góc vuông, song song)

2. ĐIỂM (POINTS): (define <name> point <construction>)
   - (midpoint B C) - trung điểm
   - (centroid A B C) - trọng tâm
   - (orthocenter A B C) - trực tâm
   - (incenter A B C) - tâm nội tiếp
   - (circumcenter A B C) - tâm ngoại tiếp
   - (projection A (segment B C)) - hình chiếu/đường cao
   - (bisector B A C) - phân giác góc BAC từ đỉnh A
   - (segment A B) - điểm trên đoạn thẳng
   - (line A B) - điểm trên đường thẳng

3. ĐOẠN/ĐƯỜNG:
   - (segment A B) - đoạn thẳng (hữu hạn)
   - (line A B) - đường thẳng (vô hạn)

4. ĐƯỜNG TRÒN: (circle <center> <type>)
   - (incircle A B C) - nội tiếp tam giác
   - (circumcircle A B C) - ngoại tiếp tam giác
   - (incircle A B C D) - nội tiếp hình vuông
   - (circumcircle A B C D) - ngoại tiếp hình vuông
   LUÔN khai báo CẢ tâm (define point) VÀ đường tròn (circle)

5. RÀNG BUỘC:
   - (parallel (segment B C) (segment D E))
   - (perpendicular (segment A B) (segment C D))
   - (angle-equal A B C D E F) - ∠ABC = ∠DEF
   Khai báo segment/line TRƯỚC khi dùng ràng buộc

═══ TỪ KHÓA TIẾNG VIỆT → DSL ═══
- trung điểm → midpoint
- trọng tâm → centroid
- trực tâm → orthocenter
- tâm nội tiếp/đường tròn nội tiếp → incenter/incircle
- tâm ngoại tiếp/đường tròn ngoại tiếp → circumcenter/circumcircle
- đường cao/chân đường cao/hình chiếu → projection
- cân tại → isosceles
- vuông tại → right
- đều → equilateral
- tù → obtuse
- hình vuông → square
- tâm hình vuông → midpoint (của đường chéo)
- đường chéo → segment
- song song → parallel
- vuông góc → perpendicular
- phân giác/đường phân giác -> bisector
- góc bằng nhau/hai góc bằng nhau → angle-equal


═══ QUY TẮC QUAN TRỌNG ═══

1. THỨ TỰ KHAI BÁO:
   ① Hình (triangle/square) - LUÔN TRƯỚC
   ② Define points
   ③ Segments/Lines
   ④ Circles
   ⑤ Constraints (parallel/perpendicular)

2. TRƯỜNG HỢP ĐẶC BIỆT:
   • Trung tuyến AM: (define M point (midpoint B C)) + (segment A M)
   • Đường cao AH: (define H point (projection A (segment B C))) + (segment A H)
   • Phân giác AD: (define D point (bisector B A C)) + (segment A D)
   • Đường thẳng vuông góc qua C: (define H point (projection C (segment A B))) + (line C H)
   • Đường trung bình DE: (define D point (midpoint A B)) + (define E point (midpoint A C)) + (segment D E)
   • Tâm hình vuông O: (define O point (midpoint A C))
   • Đường chéo hình vuông: (segment A C) và/hoặc (segment B D)

3. HÌNH VUÔNG:
   • "Hình vuông ABCD" / "góc vuông" / "cạnh bằng nhau" → CHỈ (square (A B C D))
   • CHỈ thêm khai báo khi đề bài yêu cầu: tâm, đường chéo, trung điểm, đường tròn

4. ĐƯỜNG TRÒN:
   • LUÔN: (define tâm point ...) TRƯỚC → (circle tâm ...) SAU
   • Tam giác: incircle/circumcircle với 3 điểm
   • Hình vuông: incircle/circumcircle với 4 điểm

═══ VÍ DỤ ═══

1. Tam giác cơ bản:
   "Tam giác ABC, M là trung điểm BC"
   → (triangle (A B C))
(define M point (midpoint B C))
(segment A M)

2. Tam giác với đường tròn:
   "Tam giác ABC vuông tại B, đường tròn nội tiếp O"
   → (triangle (A B C) (right B))
(define O point (incenter A B C))
(circle O (incircle A B C))

3. Tam giác cân với đường cao:
   "Tam giác ABC cân tại A, đường cao AH"
   → (triangle (A B C) (isosceles A))
(define H point (projection A (segment B C)))
(segment A H)

4. Tam giác tù:
   "Tam giác ABC tù tại B"
   → (triangle (A B C) (obtuse B))

   "Tam giác ABC có góc B tù, M là trung điểm BC"
   → (triangle (A B C) (obtuse B))
(define M point (midpoint B C))
(segment A M)

5. "Tam giác ABC có AD là đường phân giác của góc A"
   → (triangle (A B C))
(define D point (bisector B A C))
(segment A D)

TRƯỜNG HỢP KHÁC:
"Tam giác ABC có góc BAD bằng góc CAD, M là trung điểm của BC."
   → (triangle (A B C))
(define M point (midpoint B C))
(angle-equal B A D C A D)
(segment A D) 

6. Tam giác với song song:
   "Tam giác ABC, D trên AB, E trên AC, BC // DE"
   → (triangle (A B C))
(define D point (segment A B))
(define E point (segment A C))
(segment B C)
(segment D E)
(parallel (segment B C) (segment D E))

7. Hình vuông đơn giản:
   "Hình vuông ABCD" / "Hình vuông ABCD có AB vuông góc BC"
   → (square (A B C D))

8. Hình vuông với đường chéo:
   "Hình vuông ABCD, hai đường chéo AC và BD cắt nhau tại O"
   → (square (A B C D))
(define O point (midpoint A C))
(segment A C)
(segment B D)

9. Hình vuông với đường tròn:
   "Hình vuông ABCD nội tiếp đường tròn"
   → (square (A B C D))
(define O point (midpoint A C))
(circle O (circumcircle A B C D))

10. Tam giác với góc bằng nhau:
   "Tam giác ABC có góc BAD bằng góc CAD"
   → (triangle (A B C))
(define D point (segment B C))
(angle-equal B A D C A D)

═══ OUTPUT FORMAT ═══
CHỈ trả về JSON array:
[{"instruction": "Tiếng Việt", "answer": "DSL với \\n"}]

Không markdown, không giải thích.

═══ INPUT ═══
{{ extract }}

"""