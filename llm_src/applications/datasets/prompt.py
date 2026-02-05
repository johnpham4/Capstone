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
   
   LƯU Ý: Trong cú pháp angle/bisector, đỉnh góc luôn nằm Ở GIỮA:
     • (bisector B A C) = phân giác góc BAC, đỉnh tại A
     • (angle-measure B A C 80) = góc BAC = 80°, đỉnh tại A

3. ĐOẠN/ĐƯỜNG:
   - (segment A B) - đoạn thẳng (hữu hạn)
   - (line A B) - đường thẳng (vô hạn)

4. ĐƯỜNG TRÒN: (circle <center> <type>)
   - (incircle A B C) - nội tiếp tam giác
   - (circumcircle A B C) - ngoại tiếp tam giác
   - (incircle A B C D) - nội tiếp hình vuông
   - (circumcircle A B C D) - ngoại tiếp hình vuông
   - (circle O) - đường tròn tâm O (bán kính mặc định)
   - (circle O (radius 0.5)) - đường tròn tâm O, bán kính 5cm (scale: 1cm = 0.1 đơn vị)
   
   LUÔN khai báo CẢ tâm (define point) VÀ đường tròn (circle)
   
   ⚠️ QUAN TRỌNG - CÚ PHÁP CIRCLE:
   • Nếu đề BÀI CHO bán kính cụ thể → dùng: (circle O (radius 0.5))
   • Nếu đề KHÔNG NÓI bán kính → dùng: (circle O)
   • GỌN HƠN: Không cần (radius ...) khi không biết bán kính
   
   LƯU Ý SCALE: 1 cm = 0.1 đơn vị tọa độ
   **QUY ĐỔI: cm × 0.1 = giá trị DSL**
   - 4 cm → 0.4
   - 5 cm → 0.5
   - 10 cm → 1.0
   - 50 cm → 5.0
   - √15 cm ≈ 3.87 cm → 0.387
   
   DÂY CỦA ĐƯỜNG TRÒN:
   Dây là đoạn thẳng nối hai điểm bất kỳ trên đường tròn.
   
   ⚠️ QUAN TRỌNG: Khi đề bài nói "dây AB" → BẮT BUỘC phải có:
   1. (segment A B) - khai báo đoạn thẳng
   2. (on-circle A O) - A nằm trên đường tròn
   3. (on-circle B O) - B nằm trên đường tròn
   KHÔNG được thiếu (on-circle) vì không có constraint này thì A, B không nằm trên đường tròn!
   
   Ví dụ 1: "đường tròn (O) bán kính 5 cm, dây AB. Gọi H là trung điểm của AB"
   → (define O point)
      (circle O (radius 0.5))
      (define A point)
      (define B point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (define H point (midpoint A B))
   
   Ví dụ 2: "đường tròn (O) bán kính 6 cm, dây AB sao cho góc ở tâm AOB = 120°"
   → (define O point)
      (circle O (radius 0.6))
      (define A point)
      (define B point)
      (segment O A)
      (segment O B)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (angle-measure A O B 120)
   
   Ví dụ 3: "đường tròn (O) bán kính 5 cm, M là điểm bất kỳ trên đường tròn"
   → (define O point)
      (circle O (radius 0.5))
      (define M point)
      (on-circle M O)
   
   Ví dụ 4: "đường tròn (O), dây AB, M là điểm trên đường tròn. Nối MA, MB"
   → (define O point)
      (circle O)
      (define A point)
      (define B point)
      (define M point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (on-circle M O)
      (segment M A)
      (segment M B)
   
   Ví dụ 5: "đường tròn (O) bán kính 6 cm, dây AB không đi qua tâm. M là điểm trên đường tròn, 
             H là giao điểm của OM với AB, biết OM vuông góc với AB"
   → (define O point)
      (circle O (radius 0.6))
      (define A point)
      (define B point)
      (define M point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (on-circle M O)
      (define H point (midpoint A B))
      (segment O M)
      (on-segment H O M)
      (perpendicular (segment O M) (segment A B))
   
   LƯU Ý TÍNH CHẤT ĐƯỜNG TRÒN:
   • Khi "dây AB KHÔNG ĐI QUA TÂM" → KHÔNG thêm (on-segment O A B)
     Optimizer sẽ tự động tìm vị trí A, B không thẳng hàng với O
   • Khi "OM vuông góc với dây AB tại H" → H LÀ TRUNG ĐIỂM của AB (tính chất hình học)
     Biểu diễn: (define H point (midpoint A B)) + (on-segment H O M) + (perpendicular ...)
   • Khi "qua M kẻ dây AB" hoặc "dây AB đi qua M":
     → Chỉ suy ra: A, M, B thẳng hàng + A, B trên đường tròn
     → KHÔNG tự động suy ra M nằm giữa A và B
     → Biểu diễn: CHỈ khai báo dây AB với (segment A B), (on-circle A O), (on-circle B O)
     → KHÔNG thêm (on-segment M A B) trừ khi đề nói RÕ "M nằm giữa A và B" hoặc "M thuộc đoạn AB"
   
   ĐƯỜNG KÍNH:
   Đường kính là dây đặc biệt đi qua tâm O, có độ dài bằng 2 × bán kính.
   
   Ví dụ 6: "đường tròn (O) đường kính AB"
   → (define O point)
      (circle O)
      (define A point)
      (define B point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (on-segment O A B)
   
   Ví dụ 7: "đường tròn (O) đường kính AB; C là điểm trên đường tròn; H là hình chiếu của C lên AB"
   → (define O point)
      (circle O)
      (define A point)
      (define B point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (on-segment O A B)
      (define C point)
      (on-circle C O)
      (define H point (projection C (segment A B)))
      (segment C H)
   
   Ví dụ 8: "đường tròn (O) đường kính AB; C trên đường tròn; H là hình chiếu C lên AB.
             Trên OC lấy M sao cho OM = OH"
   → (define O point)
      (circle O)
      (define A point)
      (define B point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (on-segment O A B)
      (define C point)
      (on-circle C O)
      (define H point (projection C (segment A B)))
      (segment C H)
      (segment O C)
      (define M point)
      (on-segment M O C)
      (equal-distance O M O H)
   
   Ví dụ 9: "đường tròn (O) bán kính 7 cm, điểm M nằm trong đường tròn sao cho OM = 3 cm.
             Qua M kẻ dây AB của đường tròn (O)"
   → (define O point)
      (circle O (radius 0.7))
      (define M point)
      (distance O M 0.3)
      (define A point)
      (define B point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
   
   Ví dụ 10: "đường tròn (O) đường kính AB; C trên đường tròn; H là hình chiếu C lên AB.
              Trên đoạn AC lấy điểm M sao cho CM = CH"
   → (define O point)
      (circle O)
      (define A point)
      (define B point)
      (segment A B)
      (on-circle A O)
      (on-circle B O)
      (on-segment O A B)
      (define C point)
      (on-circle C O)
      (define H point (projection C (segment A B)))
      (segment C H)
      (segment A C)
      (define M point)
      (on-segment M A C)
      (equal-distance C M C H)
   
   ❌ SAI - KHÔNG dùng (on-circle ...) trong (define point ...):
   (define A point (on-circle A O))
   
   ❌ SAI - KHÔNG dùng tên điểm trong định nghĩa chính nó:
   (define A point (segment M A))
   (define B point (segment M B))
   
   ❌ SAI - Thiếu constraint on-circle:
   (segment A B)
   (define H point (midpoint A B))
   
   ❌ SAI - Tự động thêm on-segment khi đề chỉ nói "qua M kẻ dây AB":
   (on-segment M A B)  ← CHỈ dùng khi đề nói RÕ "M nằm giữa" hoặc "M thuộc đoạn AB"
   
   LƯU Ý:
   - KHI ĐỀ BÀI NÓI "DÂY AB" → LUÔN LUÔN thêm (on-circle A O) VÀ (on-circle B O)
   - KHI ĐỀ BÀI NÓI "QUA M KẺ DÂY AB" → thêm (on-segment M A B) - M nằm trên dây AB
   - Nếu đề bài nhắc đến GÓC Ở TÂM → phải vẽ CẢ 2 BÁN KÍNH (segment O A) và (segment O B)
   - Đường kính là dây đi qua tâm → thêm (on-segment O A B)

5. RÀNG BUỘC:
   - (parallel (segment B C) (segment D E))
   - (perpendicular (segment A B) (segment C D))
   - (angle-equal A B C D E F) - ∠ABC = ∠DEF
   - (on-segment M C D) - điểm M nằm trên đoạn thẳng CD
   - (distance O A 0.387) - khoảng cách OA = 0.387 đơn vị (3.87 cm)
   - (equal-distance O M O H) - khoảng cách OM = khoảng cách OH
   - (on-circle A O) - điểm A nằm trên đường tròn TÂM O
   
     LUÔN chỉ rõ TÂM để phân biệt khi có nhiều đường tròn
   
   Khai báo segment/line/circle TRƯỚC khi dùng ràng buộc
   
   LƯU Ý SCALE khoảng cách: 1 cm = 0.1 đơn vị
   **QUY ĐỔI: cm × 0.1 = giá trị DSL**
   - 3 cm → 0.3
   - 4 cm → 0.4
   - 5 cm → 0.5
   - 10 cm → 1.0
   - 50 cm → 5.0
   - √15 cm ≈ 3.87 cm → 0.387

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
- số đo góc → angle-measure
- nằm trên/nằm trên đoạn thẳng → on-segment
- đường tròn tâm O (không nói bán kính) → circle O
- đường tròn tâm O bán kính R → circle O (radius R_scaled)
- nằm trên đường tròn → on-circle
- dây (của đường tròn) → segment + on-circle (cho cả 2 điểm)
- đường kính → segment + on-circle (cả 2 điểm) + on-segment (tâm nằm giữa)
- nằm ngoài đường tròn → distance với giá trị > radius
- cắt đường tròn tại → define point (line ...) + on-circle
- nằm giữa → on-segment
- khoảng cách/cách → distance
- khoảng cách ... bằng khoảng cách .../hai khoảng cách bằng nhau → equal-distance


═══ QUY TẮC QUAN TRỌNG ═══

0. QUY TẮC VÀNG - ĐỌC KỸ TRƯỚC KHI VIẾT DSL:
   
   **Khi đề bài nói "góc X" (X là tên điểm):**
   → Nghĩa là góc TẠI ĐỈNH X
   → Trong DSL: vertex X phải Ở GIỮA
   
   VÍ DỤ BẮT BUỘC PHẢI NHỚ:
    "góc A = 60°" → (angle-measure B A C 60)   [A ở GIỮA]
    "góc B = 90°" → (angle-measure A B C 90)   [B ở GIỮA]
    "góc C = 120°" → (angle-measure A C B 120) [C ở GIỮA]
   
   SAI: "góc A = 60°" → (angle-measure A B C 60)  [B ở giữa → góc B!]
   SAI: "góc B = 90°" → (angle-measure B A C 90)  [A ở giữa → góc A!]

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

6. "Đường phân giác góc BAC cắt BC tại D"
   → (triangle (A B C))
(define D point (bisector B A C))
(segment A D)

7. "Phân giác của góc BAC cắt cạnh BC tại điểm D"
   → (triangle (A B C))
(define D point (bisector B A C))
(segment A D)

8. "Tam giác ABC, CD là trung tuyến cạnh AB, AM là phân giác góc A với M nằm trên BC"
   → (triangle (A B C))
(define D point (midpoint A B))
(segment C D)
(define M point (bisector B A C))
(segment A M)

8b. "Tam giác ABC tù tại C, góc ACB = 110°, CD là trung tuyến AB, AM là phân giác góc A với M nằm trên CD"
   → (triangle (A B C) (obtuse C))
(angle-measure A C B 110)
(define D point (midpoint A B))
(segment C D)
(define M point (bisector B A C))
(on-segment M C D)
(segment A M)

9. "Tam giác ABC tù tại B, góc ABC = 120 độ"
   → (triangle (A B C) (obtuse B))
(angle-measure A B C 120)

9b. "Tam giác ABC, góc A = 80 độ"
   → (triangle (A B C))
(angle-measure B A C 80)

9c. "Tam giác ABC vuông tại B, góc A = 40 độ"
   → (triangle (A B C) (right B))
(angle-measure B A C 40)
   SAI: (angle-measure A B C 40)  [Đây là góc B, không phải góc A!]

9d. "Tam giác ABC, góc B = 50 độ, góc C = 60 độ"
   → (triangle (A B C))
(angle-measure A B C 50)
(angle-measure A C B 60)

LƯU Ý: "góc A" = góc tại đỉnh A = góc BAC (vertex ở giữa cú pháp)

10. "Tam giác vuông ABC vuông tại A, góc BAC = 90 độ, góc ABC = 30 độ"
   → (triangle (A B C) (right A))
(angle-measure B A C 90)
(angle-measure A B C 30)

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

11. Đường tròn với điểm ở vị trí cụ thể:
   "Cho đường tròn (O; 4 cm) và hai điểm A, B. Biết rằng OA = √15 cm và OB = 4 cm"
   → (define O point)
(circle O (radius 0.4))
(define A point)
(define B point)
(distance O A 0.387)
(on-circle B O)

   LƯU Ý: 4 cm × 0.1 → 0.4, √15 ≈ 3.87 cm × 0.1 → 0.387

12. Đường tròn với điểm bên trong và trên đường tròn:
   "Đường tròn tâm O bán kính 5 cm, điểm A cách O là 3 cm, điểm B nằm trên đường tròn"
   → (define O point)
(circle O (radius 0.5))
(define A point)
(define B point)
(distance O A 0.3)
(on-circle B O)

   LƯU Ý: 5 cm × 0.1 → 0.5, 3 cm × 0.1 → 0.3

13. Đường thẳng cắt đường tròn:
   "Đường tròn (O) bán kính 5 cm, điểm M cách O 15 cm. Tia MO cắt (O) tại A, B (A nằm giữa M và O)"
   → (define O point)
(circle O (radius 0.5))
(define M point)
(distance O M 1.5)
(line M O)
(define A point (line M O))
(define B point (line M O))
(on-circle A O)
(on-circle B O)
(on-segment A M O)

   LƯU Ý: 
   - M nằm ngoài (O): OM = 15cm > radius = 5cm
   - Đường thẳng MO cắt (O) tại 2 điểm A, B
   - A nằm giữa M và O: dùng (on-segment A M O)
   - Scale: 5cm × 0.1 → 0.5, 15cm × 0.1 → 1.5

═══ OUTPUT FORMAT ═══
⚠️ QUAN TRỌNG - ĐỌC KỚ TRƯỚC KHI TẠO OUTPUT:

1. CHỈ trả về JSON array:
   [{"instruction": "Tiếng Việt", "answer": "DSL với \\n"}]

2. TUYỆT ĐỐI KHÔNG được có comment # trong field "answer":
   ❌ SAI: "(segment A B)                     # Dây AB"
   ✅ ĐÚNG: "(segment A B)"
   
3. Field "answer" CHỈ chứa DSL thuần túy, không giải thích, không comment

4. Không markdown, không giải thích bên ngoài JSON

═══ INPUT ═══
{{ extract }}

"""