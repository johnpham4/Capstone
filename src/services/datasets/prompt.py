prompt: str = """Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL (S-expression syntax).

LƯU Ý: Tên điểm trong ví dụ (A, B, C, M, O...) chỉ minh họa. PHẢI dùng đúng tên điểm trong đề bài!

═══ LỖI CẤM TUYỆT ĐỐI ═══

1. (on-segment A B) chỉ 2 điểm → SAI! Cần đúng 3: (on-segment M A B)
   Tiếp điểm là endpoint → KHÔNG dùng on-segment

2. Define trùng → SAI! Mỗi điểm CHỈ define MỘT LẦN duy nhất

3. Tiếp điểm thiếu on-circle → SAI!
   Phải: (define T point) → (on-circle T O) → (tangent T (circle O) AB)

4. Dây cung thiếu on-circle → SAI! Cả 2 đầu phải on-circle
   (define C point) → (on-circle C O) → (define D point) → (on-circle D O) → (segment C D)

5. Phải vẽ TẤT CẢ mô tả hình học trong đề, kể cả phần "chứng minh"
   • "Chứng minh AB = AC" → (equal-distance A B A C)
   • "Chứng minh góc A = 60°" → (angle-measure B A C 60)
   • "Chứng minh AB ⊥ CD" → (perpendicular (segment A B) (segment C D))
   • "Chứng minh ∠BAC = ∠OCA" → (angle-equal B A C O C A)

6. Mệnh đề trung điểm phải dịch đúng:
   • "H là trung điểm của AC" → (equal-distance A H H C)
   • CẤM: (equal-distance A H A C)  ← SAI nghĩa

═══ QUY TẮC VÀNG - VẼ SEGMENT KHI ĐỀ NHẮC ═══

Đọc kỹ đề → tìm TẤT CẢ cạnh/đoạn được nhắc → kiểm tra đã có segment chưa → thêm nếu thiếu
• "AC = AD" → nhắc AC, AD
• "AB ⊥ CD" → nhắc AB, CD
• "AO là đường trung trực BC" → nhắc AO, BC
• "kẻ OH", "nối AC" → nhắc OH, AC
• "đường kính MN" → nhắc MN

Đặc biệt:
• Tiếp tuyến AB → đã có (segment A B) rồi
• Dây CD → đã có (segment C D) rồi
• Đường kính MN → (diameter M N O) KHÔNG tự vẽ → PHẢI thêm (segment M N)

BÁN KÍNH (segment O X):
KHÔNG tự ý vẽ (segment O X) cho mọi điểm trên đường tròn. CHỈ vẽ khi:
• Đề nhắc trực tiếp đến OX ("bán kính OA", "đoạn OA", "nối OA")
• Cần cho góc ở tâm (∠AOB → cần OA và OB)
• Ngay sau (tangent X ...) → yêu cầu kỹ thuật optimizer
• X nằm trên đường kính đã vẽ → KHÔNG vẽ thêm

═══ CÚ PHÁP DSL ═══

1. HÌNH (SHAPES)
   • Tam giác: (triangle (A B C) [type])
     - Thường: (triangle (A B C))
     - Cân: (triangle (A B C) (isosceles A))
     - Vuông: (triangle (A B C) (right B))
     - Vuông cân: (triangle (A B C) (right_isosceles B))
     - Đều: (triangle (A B C) (equilateral))

   • Hình vuông: (square (A B C D))
   • Hình chữ nhật: (rectangle (A B C D))
   • Hình thang: (trapezoid (A B C D))
   • Hình bình hành: (parallelogram (A B C D))
   • Hình thoi: (rhombus (A B C D))
   CHỈ khai báo hình, KHÔNG assert thuộc tính tự nhiên (cạnh bằng, góc vuông, song song)

2. ĐIỂM (POINTS): (define <name> point <construction>)
   Đặc biệt:
   - (midpoint B C) - trung điểm
   - (centroid A B C) - trọng tâm
   - (orthocenter A B C) - trực tâm
   - (incenter A B C) - tâm nội tiếp
   - (circumcenter A B C) - tâm ngoại tiếp
   - (projection A (segment B C)) - hình chiếu/đường cao
   - (bisector B A C) - phân giác góc BAC từ đỉnh A
   - (inter-ll C D A B) - giao điểm của CD và AB
   Tự do:
   - (segment A B) - điểm trên đoạn thẳng
   - (line A B) - điểm trên đường thẳng

   HÌNH CHIẾU (PROJECTION):
   Khi "kẻ OH ⊥ AC (H ∈ AC)" / "H là hình chiếu O lên AC" / "OH ⊥ AC, H nằm trên AC":
   → DÙNG: (define H point (projection O (segment A C)))
   → KHÔNG dùng: (define H point) + (perpendicular ...) + (on-segment H A C)

   Projection tự động bao gồm:
   • H nằm trên AC
   • OH ⊥ AC
   → KHÔNG thêm (perpendicular (segment O H) (segment A C)) - THỪA, gây xung đột!
   → KHÔNG vẽ (segment A H) hoặc (segment H C) - VẼ ĐÈ!
   → CHỈ vẽ: (segment A C) + (define H ...) + (segment O H)

   Segment vẽ từ ĐIỂM GỐC đến HÌNH CHIẾU:
   • "H là hình chiếu O lên AC" → (segment O H)
   • "H là hình chiếu B lên AC" → (segment B H)

   Nếu đề nói "H là trung điểm AC" sau projection:
   → H ĐÃ được xác định bởi projection, KHÔNG define thêm midpoint
   → Thêm ràng buộc đúng: (equal-distance A H H C)

   CẤM dùng on-circle trong construction:
   SAI: (define C point (on-circle C O))
   ĐÚNG: (define C point) + (on-circle C O)

   TRUNG ĐIỂM CUNG vs TRUNG ĐIỂM ĐOẠN:
   • Trung điểm CUNG → điểm tự do trên đường tròn: (define C point) + (on-circle C O)
   • Trung điểm ĐOẠN → dùng midpoint: (define M point (midpoint A B))

   ĐƯỜNG TRUNG TRỰC:
   "AO là đường trung trực BC": AO đi qua trung điểm BC và AO ⊥ BC
   → CHỈ vẽ: (segment B C) + (segment A O) + (perpendicular (segment A O) (segment B C))
   → KHÔNG tự ý define điểm giao nếu đề không nhắc
   → KHÔNG vẽ thêm (segment A M) hoặc (segment M O) - VẼ ĐÈ!

   Nếu đề nhắc điểm giao (vd "M là giao điểm AO và BC"):
   → PHẢI dùng: (define M point (inter-ll A O B C))
   → KHÔNG dùng: (define M point (midpoint B C)) - chấm sẽ lệch khỏi AO!

   CHỈ DEFINE ĐIỂM KHI ĐỀ NÓI RÕ:
   • Đề có "Gọi X", "X là giao điểm", "điểm X" → DEFINE
   • Đề KHÔNG nhắc tên điểm → KHÔNG DEFINE

   ĐIỂM NẰM TRÊN ĐƯỜNG TRÒN:
   • "M ∈ (O)" / "M nằm trên (O)" → (on-circle M O) ngay sau define M
   • "(D ∈ (O))" trong đề → PHẢI có (on-circle D O)

   QUY TẮC MIDPOINT/PROJECTION:
   Segment phải tồn tại TRƯỚC khi dùng midpoint/projection
   • (triangle (A B C)) → BC đã có → (define M point (midpoint B C)) OK
   • B, C riêng lẻ → thêm (segment B C) trước midpoint

3. ĐOẠN/ĐƯỜNG:
   - (segment A B) - đoạn thẳng (hữu hạn)
   - (line A B) - đường thẳng (vô hạn)

4. ĐƯỜNG TRÒN (CIRCLES): (circle <center> <type>)
   - (circle O) hoặc (circle O (radius 0.5))
   - (incircle A B C) / (circumcircle A B C)
   - LUÔN: (define O point) TRƯỚC → (circle O) SAU
   - Scale: 1 cm = 0.1 đơn vị (5cm → 0.5, 10cm → 1.0)

   ĐƯỜNG KÍNH: (diameter A B O)
   • A, B: 2 đầu mút, O: tâm (tự động nằm giữa)
   • Tự động tạo: on-circle A O, on-circle B O, on-segment O A B
   • (diameter ...) CHỈ là constraint, KHÔNG tự vẽ → PHẢI thêm (segment A B)

   DÂY THƯỜNG (3 bước bắt buộc):
   1. Define điểm: (define C point) + (define D point)
   2. Đặt trên đường tròn: (on-circle C O) + (on-circle D O)
   3. Vẽ đoạn: (segment C D)

5. RÀNG BUỘC (CONSTRAINTS):
   - (parallel (segment B C) (segment D E))
   - (perpendicular (segment A B) (segment C D))
   - (tangent T (circle O) AB) - tiếp tuyến AB với (O) tại điểm T
   - (angle-equal A B C D E F) - ∠ABC = ∠DEF
   - (angle-measure B A C 60) - ∠BAC = 60° (đỉnh A ở GIỮA)
   - (on-segment M C D) - M nằm trên đoạn CD (cần đúng 3 điểm khác nhau)
   - (on-circle A O) - A trên đường tròn tâm O
   - (distance O A 0.5), (equal-distance O M O H)
   - (diameter A B O) - AB là đường kính qua tâm O
   Khai báo segment/line TRƯỚC khi dùng ràng buộc

   TIẾP TUYẾN:
   Cú pháp: (tangent T (circle O) XY)
   • T = tiếp điểm (tên bất kỳ), (circle O) = đường tròn, XY = 2 ký tự tạo đường tiếp tuyến (ví dụ AB, AT)

   Quy trình vẽ tiếp tuyến:
   1. (define T point) + (on-circle T O)
   2. (define A point) [+ constraint nếu cần]
   3. Vẽ đoạn chứa tiếp điểm T, ví dụ: (segment A T) hoặc (segment A B)
   4. Dùng đúng 2 đầu mút của đoạn đó trong tangent, ví dụ:
      • nếu có (segment A T) → (tangent T (circle O) AT)
      • nếu có (segment A B) → (tangent T (circle O) AB)
   5. (segment O T) - bán kính NGAY SAU tangent, KHÔNG chen dòng nào!
      Ngoại lệ: T nằm trên đường kính đã vẽ → bỏ qua (segment O T)

   Tiếp điểm là ENDPOINT (90% trường hợp): KHÔNG dùng on-segment
   Tiếp điểm nằm GIỮA A và B: CẦN (on-segment T A B)

   Constraint cho điểm thứ 2 trên tiếp tuyến:
   • Đề KHÔNG nói gì về B → PHẢI thêm (equal-distance A B 1.0) để ổn định
   • Đề nói rõ về B ("B nằm trên...", "AB = ...") → theo đề

   "A nằm ngoài đường tròn" → PHẢI thêm (distance O A <value>) với value > radius
   Nếu không có bán kính cụ thể, dùng 1.5

   GÓC:
   • ∠BAC: đỉnh A ở GIỮA → (angle-measure B A C ...)
   • Góc ở tâm (đỉnh = tâm O): PHẢI vẽ bán kính đến các điểm chưa nối
     - AB là đường kính: O đã nối A, B → chỉ vẽ bán kính mới
     - A, B tự do: cần cả (segment O A) và (segment O B)

═══ TỪ KHÓA TIẾNG VIỆT → DSL ═══
- trung điểm → midpoint
- trọng tâm → centroid
- trực tâm → orthocenter
- tâm nội tiếp / đường tròn nội tiếp → incenter / incircle
- tâm ngoại tiếp / đường tròn ngoại tiếp → circumcenter / circumcircle
- đường cao / chân đường cao / hình chiếu → projection
- cân tại → isosceles
- vuông tại → right
- đều → equilateral
- hình vuông → square
- tâm hình vuông → midpoint (của đường chéo)
- đường chéo → segment
- song song → parallel
- vuông góc → perpendicular
- phân giác / đường phân giác → bisector
- góc bằng nhau → angle-equal
- đường kính → diameter
- dây cung → on-circle + segment
- tiếp tuyến → tangent
- giao điểm → inter-ll

═══ QUY TẮC QUAN TRỌNG ═══

1. THỨ TỰ KHAI BÁO:
    • Bài đa giác (triangle/square/...):
       Hình → Define points → Segments/Lines cần thiết → Constraints
    • Bài có đường tròn:
       Define tâm O → (circle O) sớm → Define các điểm còn lại → Segments/Lines → Constraints

2. TRƯỜNG HỢP ĐẶC BIỆT:
   • Trung tuyến AM: (define M point (midpoint B C)) + (segment A M)
   • Đường cao AH: (define H point (projection A (segment B C))) + (segment A H)
   • Phân giác AD: (define D point (bisector B A C)) + (segment A D)
   • Đường thẳng vuông góc qua C: (define H point (projection C (segment A B))) + (line C H)
   • Đường trung bình DE: (define D point (midpoint A B)) + (define E point (midpoint A C)) + (segment D E)
   • Tâm hình vuông O: (define O point (midpoint A C))
   • Đường chéo hình vuông: (segment A C) và/hoặc (segment B D)

3. HÌNH VUÔNG:
   • "Hình vuông ABCD" → CHỈ (square (A B C D))
   • CHỈ thêm khai báo khi đề yêu cầu: tâm, đường chéo, trung điểm, đường tròn

4. ĐƯỜNG TRÒN:
   • LUÔN: (define tâm point ...) TRƯỚC → (circle tâm ...) SAU
   • Tam giác: incircle/circumcircle với 3 điểm
   • Hình vuông: incircle/circumcircle với 4 điểm

5. TIẾP TUYẾN - THỨ TỰ BẮT BUỘC:
   Sau (tangent X ...) PHẢI có (segment O X) NGAY dòng tiếp theo
   Ngoại lệ: X nằm trên đường kính → bỏ qua
   Perpendicular với tiếp tuyến: dùng segment TIẾP TUYẾN, KHÔNG dùng bán kính
   • ĐÚNG: (perpendicular (segment M A) (segment M N)) - MA là tiếp tuyến
   • SAI: (perpendicular (segment O M) (segment M N)) - OM là bán kính!

6. VẼ TẤT CẢ MÔ TẢ HÌNH HỌC:
   Dù "cho" hay "chứng minh" → đều vẽ constraint
   • "Chứng minh AB = AC" → (equal-distance A B A C)
   • "Chứng minh ∠BAC = 60°" → (angle-measure B A C 60)
   • "Chứng minh AB ⊥ CD" → (perpendicular (segment A B) (segment C D))
   • "Chứng minh ∠AOC = ∠BAC" → (angle-equal A O C B A C)

═══ MẪU DSL TIẾP TUYẾN ═══

DẠNG 1: "AB là tiếp tuyến tại A" (A là endpoint)
(define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(equal-distance A B 1.0)
(segment A B)
(tangent A (circle O) AB)
(segment O A)

DẠNG 2: "Từ A ngoài (O) kẻ tiếp tuyến AB và AC" (2 tiếp tuyến)
(define O point)
(circle O)
(define A point)
(distance O A 1.5)
(define B point)
(on-circle B O)
(segment A B)
(tangent B (circle O) AB)
(segment O B)
(define C point)
(on-circle C O)
(segment A C)
(tangent C (circle O) AC)
(segment O C)

DẠNG 3: "Đường kính MN, tiếp tuyến tại M" (tiếp điểm trên đường kính)
(define O point)
(circle O)
(define M point)
(define N point)
(diameter M N O)
(segment M N)
(define A point)
(distance M A 1.0)
(segment M A)
(tangent M (circle O) MA)
KHÔNG vẽ (segment O M) - M đã trên đường kính!

DẠNG 4: "AB tiếp xúc tại M" (M nằm giữa A và B)
(define O point)
(circle O)
(define M point)
(on-circle M O)
(define A point)
(define B point)
(distance A B 2.0)
(segment A B)
(on-segment M A B)
(tangent M (circle O) AB)
(segment O M)

DẠNG 5: "AB là tiếp tuyến tại A, AC là dây"
(define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(equal-distance A B 1.0)
(segment A B)
(tangent A (circle O) AB)
(segment O A)
(define C point)
(on-circle C O)
(segment A C)

═══ VÍ DỤ ═══

1. Tam giác cơ bản:
   "Tam giác ABC, M là trung điểm BC"
   → (triangle (A B C))
(define M point (midpoint B C))
(segment A M)

2. Tam giác vuông với góc:
   "Tam giác ABC vuông tại B, góc A = 60°"
   → (triangle (A B C) (right B))
(angle-measure B A C 60)

3. Đường tròn với góc ở tâm:
   "Cho đường tròn (O) với ∠AOB = 120°. Lấy điểm C là trung điểm cung nhỏ AB"
   → (define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(on-circle B O)
(segment O A)
(segment O B)
(segment A B)
(angle-measure A O B 120)
(define C point)
(on-circle C O)
(segment A C)
(segment B C)

4. Đường tròn với bán kính:
   "Đường tròn (O) bán kính 6cm, dây AB, góc ở tâm AOB = 120°"
   → (define O point)
(circle O (radius 0.6))
(define A point)
(on-circle A O)
(define B point)
(on-circle B O)
(segment O A)
(segment O B)
(segment A B)
(angle-measure A O B 120)

5. Đường kính với dây vuông góc:
   "Cho đường tròn (O) đường kính AB. C trên (O), ∠AOC = 50°. Dây CD ⊥ AB (D ∈ (O))"
   → (define O point)
(circle O)
(define A point)
(define B point)
(diameter A B O)
(segment A B)
(define C point)
(on-circle C O)
(segment O C)
(angle-measure A O C 50)
(define D point)
(on-circle D O)
(segment C D)
(perpendicular (segment A B) (segment C D))

6. Đường kính với dây song song:
   "Cho đường tròn (O) đường kính AB. Dây CD ⊥ AB (D ∈ (O)) và dây DE // AB (E ∈ (O))"
   → (define O point)
(circle O)
(define A point)
(define B point)
(diameter A B O)
(segment A B)
(define C point)
(on-circle C O)
(define D point)
(on-circle D O)
(segment C D)
(perpendicular (segment A B) (segment C D))
(define E point)
(on-circle E O)
(segment D E)
(parallel (segment A B) (segment D E))

7. Hai tiếp tuyến từ điểm ngoài:
   "Từ A ngoài (O) kẻ hai tiếp tuyến AB, AC. M là trung điểm BC. Chứng minh AM ⊥ BC"
   → (define O point)
(circle O)
(define A point)
(distance O A 1.5)
(define B point)
(on-circle B O)
(segment A B)
(tangent B (circle O) AB)
(segment O B)
(define C point)
(on-circle C O)
(segment A C)
(tangent C (circle O) AC)
(segment O C)
(segment B C)
(define M point (midpoint B C))
(segment A M)
(perpendicular (segment A M) (segment B C))

8. Tiếp tuyến và dây:
   "AB là tiếp tuyến tại A, AC là dây. Chứng minh ∠BAC = ∠OCA"
   → (define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(equal-distance A B 1.0)
(segment A B)
(tangent A (circle O) AB)
(segment O A)
(define C point)
(on-circle C O)
(segment A C)
(segment O C)
(angle-equal B A C O C A)

9. Đường kính với tiếp tuyến:
   "Đường kính MN, tiếp tuyến tại M. Chứng minh tiếp tuyến ⊥ MN"
   → (define O point)
(circle O)
(define M point)
(define N point)
(diameter M N O)
(segment M N)
(define A point)
(distance M A 1.0)
(segment M A)
(tangent M (circle O) MA)
(perpendicular (segment M A) (segment M N))

10. Tiếp tuyến và dây song song:
    "AB là tiếp tuyến tại A, dây CD // AB. Chứng minh AC = AD"
    → (define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(equal-distance A B 1.0)
(segment A B)
(tangent A (circle O) AB)
(segment O A)
(define C point)
(on-circle C O)
(define D point)
(on-circle D O)
(segment C D)
(parallel (segment A B) (segment C D))
(segment A C)
(segment A D)
(equal-distance A C A D)

11. Đường kính với hình chiếu:
    "Đường tròn (O) đường kính AB, C trên (O), H là hình chiếu C lên AB"
    → (define O point)
(circle O)
(define A point)
(define B point)
(diameter A B O)
(segment A B)
(define C point)
(on-circle C O)
(define H point (projection C (segment A B)))
(segment C H)

12. Đường kính, dây cắt nhau:
    "Đường tròn (O) đường kính MN. Dây AB cắt MN tại H"
    → (define O point)
(circle O)
(define M point)
(define N point)
(diameter M N O)
(segment M N)
(define A point)
(on-circle A O)
(define B point)
(on-circle B O)
(segment A B)
(define H point (inter-ll A B M N))

13. Tiếp tuyến tại A, AC là dây, OH ⟂ AC và H là trung điểm AC:
   "Cho đường tròn (O). AB là tiếp tuyến tại A, AC là dây. Kẻ OH vuông góc AC (H ∈ AC). Chứng minh H là trung điểm của AC."
   → (define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(equal-distance A B 1.0)
(segment A B)
(tangent A (circle O) AB)
(segment O A)
(define C point)
(on-circle C O)
(segment A C)
(define H point (projection O (segment A C)))
(segment O H)
(equal-distance A H H C)

═══ OUTPUT FORMAT ═══

1. CHỈ trả về JSON: [{"instruction": "...", "answer": "DSL với \\n"}]
2. Field "instruction" PHẢI là bài toán tiếng Việt gốc, KHÔNG thay đổi
3. Field "answer" CHỈ chứa DSL thuần túy (KHÔNG comment #, KHÔNG giải thích)
4. KHÔNG markdown, KHÔNG giải thích bên ngoài JSON

═══ CHECKLIST TRƯỚC KHI OUTPUT ═══

1. Mỗi điểm chỉ define MỘT LẦN
2. on-segment đều có đúng 3 điểm khác nhau
3. Tiếp điểm có (on-circle T O) trước (tangent T ...), với T là tên tiếp điểm bất kỳ
4. Tiếp điểm là endpoint → KHÔNG on-segment
5. "Nằm ngoài đường tròn" → có (distance O A <value>)
6. Vẽ tất cả mô tả hình học (cả "cho" và "chứng minh")
7. Điểm trên đường tròn đều có (on-circle ... O)
8. midpoint/projection → segment phải tồn tại trước
9. Dây cung → cả 2 đầu on-circle
10. Đường kính → (diameter M N O) + (segment M N)
11. Hình chiếu → dùng projection, KHÔNG define riêng + perpendicular
12. Projection đã có perpendicular → KHÔNG thêm perpendicular thừa
13. Hình chiếu: KHÔNG vẽ (segment A H) hoặc (segment H C) khi H trên AC
14. Sau (tangent X ...) → (segment O X) ngay dòng sau (trừ X trên đường kính)
15. Đường trung trực: chỉ 3 dòng nếu đề không nhắc điểm giao
16. Đường trung trực + đề nhắc điểm giao → dùng inter-ll, KHÔNG midpoint
17. Đường kính + tiếp tuyến → KHÔNG vẽ lại bán kính
18. Perpendicular tiếp tuyến → dùng segment tiếp tuyến, KHÔNG bán kính
19. Điểm thứ 2 trên tiếp tuyến không constraint từ đề → thêm (equal-distance ... 1.0)
20. Không có comment # trong DSL output
21. Field "instruction" là tiếng Việt gốc, không thay đổi
22. KHÔNG vẽ (segment O X) khi đề không nhắc - chỉ vẽ khi cần cho góc ở tâm hoặc sau tangent
23. "H là trung điểm của AC" bắt buộc là (equal-distance A H H C), không phải AH = AC

═══ INPUT ═══
{{ extract }}

"""
