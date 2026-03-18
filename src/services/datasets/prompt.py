prompt: str = """Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL (S-expression syntax).

LƯU Ý: Tên điểm trong ví dụ (A, B, C, M, O...) chỉ minh họa. PHẢI dùng đúng tên điểm trong đề bài!

═══ LỖI CẤM TUYỆT ĐỐI ═══

0. PHẠM VI TRÍCH XUẤT = TOÀN BỘ ĐỀ BÀI (không chỉ phần mô tả hình)
    • PHẢI đọc và convert cả 3 phần nếu có:
       - Giả thiết / phần "Cho..."
       - Yêu cầu / phần "Hỏi..."
       - Kết luận / phần "Chứng minh rằng..."
    • Hễ đề NHẮC tới quan hệ hình học nào thì PHẢI vẽ/constraint quan hệ đó trong DSL,
       dù nó nằm ở phần chứng minh hay phần hỏi.
    • CẤM bỏ qua chỉ vì câu nằm sau cụm "chứng minh", "hỏi", "kết luận".

    Ví dụ bắt buộc convert:
    • "Chứng minh OM ⟂ BC" → (perpendicular (segment O M) (segment B C))
    • "Chứng minh OA = OB = OC" →
       (equal-distance O A O B)
       (equal-distance O A O C)
    • "Chứng minh OM ∥ AN" → (parallel (segment O M) (segment A N))
    • "Chứng minh ID ⟂ BC" →
       (segment I D)
       (segment B C)
       (perpendicular (segment I D) (segment B C))

1. (on-segment A B) chỉ 2 điểm → SAI! Cần đúng 3: (on-segment M A B)
   Tiếp điểm là endpoint → KHÔNG dùng on-segment
   CẤM dùng on-segment trong construction của define:
   • SAI:  (define D point (on-segment A B))
   • ĐÚNG: (define D point (segment A B))

2. Define trùng → SAI! Mỗi điểm CHỈ define MỘT LẦN duy nhất

2.1. Điểm đã define bằng construction thì CẤM define lại dạng tự do:
   • Nếu đã có (define O point (inter-ll A C B D)) thì CẤM thêm (define O point)
   • Nếu đã có (define M point (midpoint B C)) thì CẤM thêm (define M point)
   • Nếu cần dùng lại điểm đã define, chỉ thêm segment/constraint, KHÔNG define lại

2.2. Đỉnh đã có từ shape thì CẤM define lại:
    • Sau (triangle (A B C)) hoặc (square/rectangle/trapezoid/parallelogram/rhombus ...),
       các đỉnh A/B/C/D đã tồn tại ngầm.
    • CẤM thêm (define A point), (define B point), (define C point), (define D point)
       trừ khi đề thật sự giới thiệu điểm mới khác tên đỉnh.

3. Tiếp điểm thiếu on-circle → SAI!
   Phải: (define T point) → (on-circle T O) → (tangent T (circle O) AB)

3.1. TIẾP TUYẾN "QUA A" (A là tiếp điểm) - QUY TẮC RIÊNG BẮT BUỘC:
   • Nếu đề nói "Qua A kẻ tiếp tuyến..." thì tiếp điểm là A (không phải điểm mới).
   • PHẢI có: (on-circle A O)
   • CẤM define lại A nếu A đã là đỉnh của tam giác/tứ giác.
   • Điểm thứ hai để tạo đường tiếp tuyến (ví dụ T) là điểm phụ trên đường thẳng tiếp tuyến,
     KHÔNG bắt buộc nằm trên đường tròn, nên CẤM thêm (on-circle T O).

   Mẫu đúng:
   (triangle (A B C))
   (define O point (circumcenter A B C))
   (circle O (circumcircle A B C))
   (define T point)
   (equal-distance A T 1.0)
   (segment A T)
   (tangent A (circle O) AT)
   (segment O A)
   (perpendicular (segment O A) (segment A T))

4. Dây cung thiếu on-circle → SAI! Cả 2 đầu phải on-circle
   (define C point) → (on-circle C O) → (define D point) → (on-circle D O) → (segment C D)

5. Phải vẽ TẤT CẢ quan hệ hình học được nêu trong toàn bộ đề (kể cả phần "hỏi/chứng minh/kết luận")
    • "Chứng minh AB = AC" → (equal-distance A B A C)
    • "Chứng minh góc BAC = 60°" / "Chứng minh ∠BAC = 60°" / "Chứng minh BAC = 60°" (ngữ cảnh là góc)
       → (angle-measure B A C 60)
    • "Chứng minh AB ⊥ CD" → (perpendicular (segment A B) (segment C D))
    • "Chứng minh góc BAC = góc OCA" / "Chứng minh ∠BAC = ∠OCA" / "Chứng minh BAC = OCA" (ngữ cảnh là góc)
       → (angle-equal B A C O C A)

5.1. Nếu phần "chứng minh" nhắc TÊN ĐOẠN (OA, OB, OC, OM, AN, ID, ...)
   thì PHẢI có dòng (segment ...) tương ứng trong DSL.
   Quy tắc tổng quát: bất kỳ đoạn XY nào được nêu trong đề (giả thiết/hỏi/chứng minh)
   đều phải có (segment X Y), không phụ thuộc tên chữ cái cụ thể.
   • "OA = OB = OC" → PHẢI có ít nhất:
      (equal-distance O A O B)
      (equal-distance O A O C)
      - BẮT BUỘC thêm (segment O A), (segment O B), (segment O C)
      - Không được bỏ các segment này chỉ vì lý do tránh chồng nét
   • "MN // AB" → BẮT BUỘC có:
      (segment M N)
      (segment A B)
      (parallel (segment M N) (segment A B))
   • "OM // AN" → BẮT BUỘC có:
      (segment O M)
      (segment A N)
      (parallel (segment O M) (segment A N))

6. Mệnh đề trung điểm phải dịch đúng:
   • "H là trung điểm của AC" → (equal-distance A H H C)
   • CẤM: (equal-distance A H A C)  ← SAI nghĩa

7. Hình thoi bắt buộc dùng rhombus:
   • ĐÚNG: (rhombus (A B C D))
   • CẤM: (diamond (A B C D))

8. CẤM ký hiệu toán học trong DSL output:
   • CẤM: ⟂, ∥, //, ∠, =
   • PHẢI dùng từ khóa DSL chữ: perpendicular, parallel, angle-measure/angle-equal, equal-distance

9. CẤM dùng equal-distance sai arity:
   • ĐÚNG: (equal-distance A C B D) hoặc (equal-distance A B 1.0)
   • SAI:  (equal-distance A C O D B)

10. "Tam giác ... là tam giác vuông" KHÔNG nêu đỉnh vuông:
   • KHÔNG tự ý thêm (angle-measure ... 90)
   • CHỈ dùng (angle-measure ... 90) khi đề ghi rõ "vuông tại <điểm>"

11. "Đường tròn nội tiếp tâm I" bắt buộc có đủ 2 dòng:
   • (define I point (incenter A B C))
   • (circle I (incircle A B C))
   • CẤM: chỉ (define I point) + (on-circle D I) khi chưa có (circle I ...)

12. "... nội tiếp đường tròn tâm X" (X là tên tâm bất kỳ) bắt buộc đặt TẤT CẢ đỉnh lên cùng đường tròn tâm X:
    • Tam giác ABC nội tiếp (X):
       - Ưu tiên: (define X point (circumcenter A B C)) + (circle X (circumcircle A B C))
       - Dù dùng circumcenter/circumcircle vẫn PHẢI ghi tường minh đủ:
          (on-circle A X)
          (on-circle B X)
          (on-circle C X)
       - CẤM chỉ có (on-circle A X) mà thiếu B/C.
    • Tứ giác ABCD nội tiếp (X) (hình vuông/chữ nhật/hình thang/...):
       - Bắt buộc có đủ: (on-circle A X), (on-circle B X), (on-circle C X), (on-circle D X)
    • CẤM chỉ viết (circle X) mà thiếu ràng buộc on-circle cho các đỉnh nội tiếp.

12.1. CỤM "nằm trên đường tròn" / "thuộc đường tròn" phải dịch trực tiếp thành on-circle:
   • "P nằm trên đường tròn tâm X" / "P ∈ (X)" → (on-circle P X)
   • "B và C nằm trên đường tròn tâm M" → (on-circle B M) + (on-circle C M)
   • CẤM đổi tâm: nếu câu nói tâm M thì không được viết (on-circle ... O)

12.2. CẤM tự sinh điểm O khi đề không nhắc O:
   • Nếu đề không chứa điểm O (không có "tâm O", "(O)", "OA/OB/OC", "... tại O"),
     thì answer KHÔNG được có bất kỳ token O nào.
   • Với bài có tâm M/N/I/... thì chỉ dùng đúng tâm đó, CẤM thêm (define O point) làm tâm mặc định.

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
• CẤM lặp lại cùng một segment đã khai báo (ví dụ đã có (segment A C) thì không khai báo lại (segment A C)/(segment C A))

BÁN KÍNH (segment O X):
KHÔNG tự ý vẽ (segment O X) cho mọi điểm trên đường tròn. CHỈ vẽ khi:
• Đề nhắc trực tiếp đến OX ("bán kính OA", "đoạn OA", "nối OA")
• Cần cho góc ở tâm (∠AOB → cần OA và OB)
• Ngay sau (tangent X ...) → yêu cầu kỹ thuật optimizer
• X nằm trên đường kính đã vẽ → KHÔNG vẽ thêm

TRÁNH ĐOẠN CON CHỒNG ĐOẠN MẸ:
• Nếu đã có (segment A C) và biết O nằm trên AC (do inter-ll/on-segment), thì KHÔNG vẽ thêm (segment O A) hoặc (segment O C)
   trừ khi đề bắt buộc nêu rõ phải "nối OA/OC".
• Nếu đã có (segment B D) và biết O nằm trên BD, thì KHÔNG vẽ thêm (segment O B) hoặc (segment O D)
   trừ khi đề bắt buộc nêu rõ phải "nối OB/OD".
• Nếu đề nhắc trực tiếp OA/OB/OC/OD (ví dụ "OA = OB = OC"), PHẢI vẽ các segment tương ứng.

ƯU TIÊN QUY TẮC:
• Nếu có (tangent X ...) thì quy tắc "vẽ (segment O X) ngay sau tangent" được ưu tiên cao hơn quy tắc "không tự vẽ bán kính".

═══ CÚ PHÁP DSL ═══

1. HÌNH (SHAPES)
   • Tam giác: (triangle (A B C) [type])
     - Thường: (triangle (A B C))
     - Cân: (triangle (A B C) (isosceles A))
     - Vuông: (triangle (A B C) (right B))
     - Vuông cân: (triangle (A B C) (right_isosceles B))
     - Đều: (triangle (A B C) (equilateral))

   • Tứ giác: (quadrilateral ABCD (A B C D))
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

   CẤM dùng on-segment trong construction:
   SAI: (define D point (on-segment A B))
   ĐÚNG: (define D point (segment A B))

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
    - LƯU Ý NĂNG LỰC DSL HIỆN TẠI: incircle/circumcircle CHỈ dùng cho 3 điểm tam giác
       • CẤM: (circumcircle A B C D), (incircle A B C D)
   - LUÔN: (define O point) TRƯỚC → (circle O) SAU
   - Scale: 1 cm = 0.1 đơn vị (5cm → 0.5, 10cm → 1.0)

    TÂM/BÁN KÍNH ĐƯỢC CHỈ ĐỊNH TRỰC TIẾP TRONG ĐỀ:
    • Nếu đề ghi rõ "đường tròn tâm X" thì PHẢI dùng đúng tâm X, KHÔNG đổi sang tâm khác.
    • Nếu đề ghi "bán kính XY" thì PHẢI encode:
       - (circle X)
       - (on-circle Y X)
       - KHÔNG dùng bán kính số tùy ý thay cho XY.
    • Ví dụ: "vẽ đường tròn tâm M bán kính MB" → (circle M) + (on-circle B M)
    • Ví dụ: "vẽ đường tròn tâm O bán kính OA" → (circle O) + (on-circle A O)

    CẤM TỰ Ý ĐỔI TÊN TÂM:
    • Đề ghi tâm M → CẤM tạo đường tròn tâm O
    • Đề ghi tâm O → CẤM tạo đường tròn tâm M/N/... 

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

    QUY TẮC CỨNG CHO PERPENDICULAR:
    • Trước mỗi dòng (perpendicular (segment X Y) (segment U V)) PHẢI có đủ:
       (segment X Y) và (segment U V)
    • Không được trông chờ renderer/optimizer tự thêm đoạn.
    • Ví dụ đề: "ID vuông góc BC" bắt buộc có:
       (segment I D)
       (segment B C)
       (perpendicular (segment I D) (segment B C))
      • Ví dụ đề: "OM vuông góc CD" bắt buộc có:
         (segment O M)
         (segment C D)
         (perpendicular (segment O M) (segment C D))

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

   Trường hợp đặc biệt "tangent tại endpoint A/B/C...":
   • Dùng chính endpoint làm tiếp điểm trong lệnh tangent, ví dụ (tangent A (circle O) AT)
   • KHÔNG đặt điểm phụ T lên đường tròn.
   • Bán kính cần vẽ là OA/OB/OC (từ tâm đến tiếp điểm endpoint), không phải OT.

   Tiếp điểm là ENDPOINT (90% trường hợp): KHÔNG dùng on-segment
   Tiếp điểm nằm GIỮA A và B: CẦN (on-segment T A B)

   Constraint cho điểm thứ 2 trên tiếp tuyến:
   • Đề KHÔNG nói gì về B → PHẢI thêm (equal-distance A B 1.0) để ổn định
   • Đề nói rõ về B ("B nằm trên...", "AB = ...") → theo đề

   "A nằm ngoài đường tròn" → PHẢI thêm (distance O A <value>) với value > radius
   Nếu không có bán kính cụ thể, dùng 1.5

      GÓC:
       • QUY TẮC TỔNG QUÁT cho mọi góc 3 ký tự XYZ:
          - "∠XYZ", "góc XYZ", hoặc chỉ "XYZ" (khi ngữ cảnh đang nói về góc) đều cùng nghĩa.
          - Chỉ coi "XYZ" là góc khi có tín hiệu ngữ cảnh góc trong cùng câu như: "góc", "số đo", "độ", "= ...°", hoặc vế còn lại là dạng góc (∠UVW / góc UVW).
       - Đỉnh luôn là chữ ở giữa: Y.
       - DSL tương ứng: (angle-measure X Y Z ...)
    • Ví dụ: ∠BAC / góc BAC → (angle-measure B A C ...)
    • Ví dụ: ∠AOC / góc AOC → (angle-measure A O C ...)
    • Với góc bằng nhau: "∠XYZ = ∠UVW" hoặc "góc XYZ bằng góc UVW" → (angle-equal X Y Z U V W)
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
       Define tâm X → (circle X) sớm → Define các điểm còn lại → Segments/Lines → Constraints

2. TRƯỜNG HỢP ĐẶC BIỆT:
   • Trung tuyến AM: (define M point (midpoint B C)) + (segment A M)
   • Đường cao AH: (define H point (projection A (segment B C))) + (segment A H)
   • Phân giác AD: (define D point (bisector B A C)) + (segment A D)
   • Đường thẳng vuông góc qua C: (define H point (projection C (segment A B))) + (line C H)
   • Đường trung bình DE: (define D point (midpoint A B)) + (define E point (midpoint A C)) + (segment D E)
   • Tâm hình vuông O: (define O point (midpoint A C))
   • Đường chéo hình vuông: (segment A C) và/hoặc (segment B D)

    • "Qua M kẻ đường thẳng song song với AB cắt AC tại D":
       - D phải nằm trên AC: (define D point (segment A C))
       - Đường qua M là MD nên song song phải là:
          (parallel (segment M D) (segment A B))
       - CẤM dùng MA để song song trong mẫu này:
          (parallel (segment M A) (segment M D))  ← SAI ngữ nghĩa

    • Mẫu tổng quát cùng cấu trúc với điểm:
       "Qua M kẻ đường thẳng song song với AB cắt AC tại N":
       - BẮT BUỘC có: (define N point (segment A C))
       - BẮT BUỘC có: (segment M N), (segment A B)
       - BẮT BUỘC có: (parallel (segment M N) (segment A B))
       - Nếu phần chứng minh còn có "OM // AN" thì thêm RIÊNG:
         (segment O M)
         (segment A N)
         (parallel (segment O M) (segment A N))

    • "Hai đường chéo AC và BD cắt nhau tại X":
       - BẮT BUỘC có: (segment A C), (segment B D)
          - BẮT BUỘC define giao điểm bằng construction:
             (define X point (inter-ll A C B D))
          - Nếu X đồng thời là tâm đường tròn, dùng lại chính X đó cho (circle X),
             KHÔNG được define X dạng tự do rồi chỉ thêm on-segment
          - CẤM encode giao điểm chỉ bằng on-segment khi đề đã nói rõ "cắt nhau tại X"
       - KHÔNG được chỉ ghi "cắt nhau tại X" mà thiếu đoạn AC/BD hoặc thiếu ràng buộc giao điểm

3. HÌNH VUÔNG:
   • "Hình vuông ABCD" → CHỈ (square (A B C D))
   • CHỈ thêm khai báo khi đề yêu cầu: tâm, đường chéo, trung điểm, đường tròn

4. ĐƯỜNG TRÒN:
   • LUÔN: (define tâm point ...) TRƯỚC → (circle tâm ...) SAU
   • Tam giác: incircle/circumcircle với 3 điểm
   • Tứ giác/hình vuông/chữ nhật: KHÔNG dùng circumcircle/incircle 4 điểm; dùng (circle tâm) + on-circle cho các đỉnh liên quan
   • Cụm "nội tiếp đường tròn tâm X": phải có on-circle cho toàn bộ đỉnh được nói là nội tiếp theo đúng tâm X
   • Nếu đề đã chỉ rõ tâm là O/M/... thì giữ nguyên đúng tên tâm đó trong toàn bộ DSL, KHÔNG tự đổi
    • Nếu đề là "vẽ đường tròn tâm M bán kính MB" thì mẫu tối thiểu phải là:
       (define M point (...))
       (circle M)
       (on-circle B M)
       CẤM đổi sang (circle O) hoặc (on-circle B O)
    • Nếu phần chứng minh nói thêm "C nằm trên đường tròn tâm M" thì PHẢI thêm (on-circle C M)

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
   • "Chứng minh C nằm trên đường tròn tâm O" → (on-circle C O)

7. MỆNH ĐỀ "THẲNG HÀNG":
   • KHÔNG được chuyển thành equal-distance tùy tiện
   • CẤM: (equal-distance A H H M) để thay cho "A,H,M thẳng hàng"
   • DSL hiện tại KHÔNG có lệnh collinear trực tiếp.
   • Chỉ encode gián tiếp khi có cấu hình chắc chắn (midpoint/projection/inter-ll...); nếu không, giữ mô tả hình học gốc và tránh bịa ràng buộc sai

8. CASE CẤM SAI - tam giác cân tại A, đường cao AH xuống BC, M là trung điểm BC:
   • CẤM sinh: (equal-distance A H H M)
   • Nếu cần encode kết luận để ổn định hình, dùng trên đáy BC:
     (equal-distance B H H C)
   • KHÔNG dùng quan hệ độ dài giữa A-H-M để biểu diễn "thẳng hàng"

9. CHECKLIST ở cuối file là checklist chính thức duy nhất trước khi output.

═══ MẪU DSL TIẾP TUYẾN ═══

MẪU 0: "Qua M kẻ đường thẳng song song với AB cắt AC tại D"
(triangle (A B C))
(define M point (midpoint B C))
(segment A B)
(segment A C)
(define D point (segment A C))
(segment M D)
(parallel (segment M D) (segment A B))
(equal-distance A D D C)

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

14. Đường tròn đường kính AB, C trên đường tròn, chứng minh tam giác ACB vuông (không nêu đỉnh):
   "Cho đường tròn tâm O có đường kính AB. Lấy điểm C nằm trên đường tròn. Chứng minh rằng tam giác ACB là tam giác vuông."
   → (define O point)
(circle O)
(define A point)
(define B point)
(diameter A B O)
(segment A B)
(define C point)
(on-circle C O)
(segment A C)
(segment B C)
   KHÔNG thêm (angle-measure ... 90) vì đề không nêu "vuông tại ..."

15. Tam giác có đường tròn nội tiếp tâm I, tiếp xúc BC/CA/AB tại D/E/F:
   "Cho tam giác ABC có đường tròn nội tiếp tâm I. Đường tròn tiếp xúc với BC tại D, CA tại E, AB tại F."
   → (triangle (A B C))
(define I point (incenter A B C))
(circle I (incircle A B C))
(segment B C)
(segment C A)
(segment A B)
(define D point (segment B C))
(on-circle D I)
(define E point (segment C A))
(on-circle E I)
(define F point (segment A B))
(on-circle F I)

16. Hình vuông ABCD, chứng minh AC = BD:
   → (square (A B C D))
(segment A C)
(segment B D)
(define O point (inter-ll A C B D))
(equal-distance A C B D)

17. Tam giác cân tại A, AH là đường cao xuống BC, M là trung điểm BC:
   "Cho tam giác ABC cân tại A. Kẻ đường cao AH xuống BC. Gọi M là trung điểm của BC. Chứng minh rằng A, H, M thẳng hàng."
   → (triangle (A B C) (isosceles A))
(segment B C)
(define H point (projection A (segment B C)))
(define M point (midpoint B C))
(segment A H)
   Tùy chọn để ổn định: (equal-distance B H H C)
   CẤM: (equal-distance A H H M)

18. Tam giác vuông tại A, M là trung điểm BC, vẽ đường tròn tâm M bán kính MB:
   "Cho tam giác ABC vuông tại A. Gọi M là trung điểm của BC. Vẽ đường tròn tâm M bán kính MB."
   → (triangle (A B C) (right A))
(segment B C)
(define M point (midpoint B C))
(circle M)
(on-circle B M)
(segment M B)
(on-circle A M)

19. Hình bình hành ABCD, AC và BD cắt nhau tại O, vẽ đường tròn tâm O bán kính OA:
   "Cho hình bình hành ABCD. Hai đường chéo AC và BD cắt nhau tại O. Vẽ đường tròn tâm O bán kính OA."
   → (parallelogram (A B C D))
(segment A C)
(segment B D)
(define O point (inter-ll A C B D))
(circle O)
(on-circle A O)
(segment O A)

20. Hình chữ nhật ABCD nội tiếp đường tròn tâm O, M là trung điểm AB, chứng minh OM // AD:
   "Cho hình chữ nhật ABCD nội tiếp đường tròn tâm O. Gọi M là trung điểm của AB. Chứng minh OM song song AD."
   → (rectangle (A B C D))
(define O point)
(circle O)
(on-circle A O)
(on-circle B O)
(on-circle C O)
(on-circle D O)
(segment A B)
(segment A D)
(define M point (midpoint A B))
(segment O M)
(parallel (segment O M) (segment A D))

21. Hình vuông ABCD nội tiếp đường tròn tâm O, chứng minh AC và BD cắt nhau tại tâm O:
   "Cho hình vuông ABCD nội tiếp đường tròn tâm O. Chứng minh rằng AC và BD cắt nhau tại tâm O."
   → (square (A B C D))
(define O point)
(circle O)
(on-circle A O)
(on-circle B O)
(on-circle C O)
(on-circle D O)
(segment A C)
(segment B D)
(on-segment O A C)
(on-segment O B D)

═══ OUTPUT FORMAT ═══

1. CHỈ trả về JSON: [{"instruction": "...", "answer": "DSL với \\n"}]
2. Field "instruction" PHẢI là bài toán tiếng Việt gốc, KHÔNG thay đổi
3. Field "answer" CHỈ chứa DSL thuần túy (KHÔNG comment #, KHÔNG giải thích)
4. KHÔNG markdown, KHÔNG giải thích bên ngoài JSON

5. MỖI dòng DSL phải là 1 S-expression hoàn chỉnh:
   • BẮT ĐẦU bằng "(" và KẾT THÚC bằng ")"
   • Ví dụ đúng: (equal-distance A M M B)
   • Ví dụ sai: equal-distance A M M B)

6. Kiểm tra ngoặc toàn cục trước khi trả output:
   • Số lượng "(" phải bằng số lượng ")"
   • Không được thiếu ngoặc ở shape line
     - ĐÚNG: (triangle (A B C) (right A))
     - SAI:  (triangle (A B C) (right A)

═══ CHECKLIST TRƯỚC KHI OUTPUT ═══

1. Mỗi điểm chỉ define MỘT LẦN
1.1. Điểm define từ midpoint/inter-ll/projection/... KHÔNG được define lại dạng tự do
2. on-segment đều có đúng 3 điểm khác nhau
2.1. CẤM dùng (on-segment ...) trong define construction; nếu cần điểm trên đoạn AB thì dùng (define X point (segment A B))
3. Nếu tiếp điểm là điểm riêng T (không phải endpoint đã có), phải có (on-circle T O) trước (tangent T ...)
4. Tiếp điểm là endpoint → KHÔNG on-segment
5. "Nằm ngoài đường tròn" → có (distance O A <value>)
6. Vẽ tất cả mô tả hình học (cả "cho" và "chứng minh")
7. Điểm trên đường tròn đều có (on-circle ... X) theo đúng tâm được khai báo trong bài
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
24. Mọi dòng answer đều phải bắt đầu bằng "(" và kết thúc bằng ")"
25. Tổng số ngoặc mở/đóng trong toàn bộ answer phải cân bằng
26. Nếu đề chỉ định tâm đường tròn (O/M/...), answer KHÔNG được đổi sang tâm khác
27. Nếu đề có "bán kính XY", answer PHẢI có (on-circle Y X)
28. Nếu đề có "đường chéo AC, BD cắt nhau tại X":
   - BẮT BUỘC có (define X point (inter-ll A C B D))
   - Không được thay thế bằng define tự do + on-segment
   - Nếu X đồng thời là tâm đường tròn thì dùng lại X đó cho (circle X)
29. Không được lặp segment trùng nhau (XY và YX tính là cùng một segment)
30. Nếu X là giao điểm trên đoạn đã vẽ (ví dụ O trên AC/BD), không vẽ thêm đoạn con XA/XB/XC/XD trừ khi đề yêu cầu tường minh
31. Nếu đề nhắc trực tiếp tên đoạn (OA/OB/OC/OD/...), PHẢI có segment tương ứng dù có nguy cơ chồng nét

═══ INPUT ═══
{{ extract }}

"""
