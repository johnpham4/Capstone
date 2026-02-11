prompt: str = """
Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL.

⚠️⚠️⚠️ 5 LỖI CẤM TUYỆT ĐỐI - ĐỌC TRƯỚC KHI LÀM ⚠️⚠️⚠️

1. ❌ (on-segment A B) - CHỈ 2 ĐIỂM → SAI!
   ✅ (on-segment M A B) - CẦN ĐÚNG 3 ĐIỂM
   • Cú pháp: (on-segment <điểm-nằm-giữa> <endpoint1> <endpoint2>)
   • VÍ DỤ: M nằm giữa A và B → (on-segment M A B)
   • ⚠️⚠️⚠️ ĐẶC BIỆT VỚI TIẾP TUYẾN:
     - "AB là tiếp tuyến tại A" → A là endpoint → KHÔNG DÙNG (on-segment A B)
     - "Từ A kẻ tiếp tuyến AM" → M là endpoint → KHÔNG DÙNG (on-segment A M)
     - "AB tiếp xúc tại M (M giữa A và B)" → CẦN (on-segment M A B)

2. ❌ (define M point) ... (define M point) - DEFINE TRÙNG → SAI!
   ✅ MỖI ĐIỂM CHỈ DEFINE MỘT LẦN DUY NHẤT
   • Nếu M đã được define → KHÔNG BAO GIỜ define M lại
   • Tìm trong toàn bộ DSL trước khi define điểm mới

3. ❌ Tiếp điểm nhưng THIẾU (on-circle M O) → SAI!
   ✅ (define M point) → (on-circle M O) → (tangent M (circle O) AB)
   • Tiếp điểm LUÔN LUÔN nằm trên đường tròn
   • Không có on-circle → M có thể rơi ra ngoài đường tròn

4. ❌ "Lấy dây CD" nhưng THIẾU (on-circle C O) và (on-circle D O) → SAI!
   ✅ Dây cung → CẢ 2 ĐẦU đều phải on-circle
   (define C point)
   (on-circle C O)  ← BẮT BUỘC!
   (define D point)
   (on-circle D O)  ← BẮT BUỘC!
   (segment C D)

5. ❌ Thêm constraint cho YÊU CẦU CHỨNG MINH → SAI!
   ✅ CHỈ thêm constraint khi đề bài CHO SẴN
   • Đề: "Chứng minh AB = AC" → KHÔNG thêm (equal-distance ...)
   • Đề: "Chứng minh góc A = 60°" → KHÔNG thêm (angle-measure ...)
   • Đề: "Cho góc A = 60°" → ĐƯỢC thêm (angle-measure B A C 60)

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
   
   ⚠️ HÌNH CHIẾU + VUÔNG GÓC - CỰC QUAN TRỌNG:
   Khi đề nói: "Kẻ OH ⊥ AC (H ∈ AC)" hoặc "H là hình chiếu O lên AC"
   → DÙNG PROJECTION, KHÔNG define H riêng + perpendicular
   
   SAI: (define H point) + (segment O H) + (on-segment H A C) + (perpendicular (segment O H) (segment A C))
   ĐÚNG: (define H point (projection O (segment A C))) + (segment O H)
   
   VÍ DỤ: "Kẻ OH ⊥ AC (H ∈ AC)"
   (segment A C)        ← Phải có segment trước!
   (define H point (projection O (segment A C)))
   (segment O H)
   
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
   
   CHỨNG MINH vs CHO SẴN:
   • Đề bảo "Chứng minh góc A = 60°" / "Tính góc A" → KHÔNG thêm (angle-measure B A C 60)
   • Đề bảo "Chứng minh OC ⟂ AB" → KHÔNG thêm (perpendicular ...)
   • CHỈ thêm constraint khi đề BÀI CHO SẴN, không phải chứng minh
   
   ĐIỂM NẰM TRÊN ĐƯỜNG TRÒN - CỰC KỲ QUAN TRỌNG:
   • Khi đề bài nói "điểm M ∈ (O)" / "M nằm trên (O)" / "M thuộc đường tròn (O)"
     → LUÔN LUÔN thêm (on-circle M O) NGAY SAU khi define M
   • Khi đề bài viết (D ∈ (O)) / (E ∈ (O)) trong ngoặc đơn
     → D và E PHẢI có (on-circle D O) và (on-circle E O)
   • VÍ DỤ: "Vẽ dây CD vuông góc với AB (D ∈ (O)) và dây DE song song với AB (E ∈ (O))"
     (define C point)
     (on-circle C O)  ← C nằm trên đường tròn
     (define D point)
     (on-circle D O)  ← THIẾU DÒNG NÀY = SAI HOÀN TOÀN!
     (segment C D)
     (perpendicular (segment A B) (segment C D))
     (define E point)
     (on-circle E O)  ← THIẾU DÒNG NÀY = SAI HOÀN TOÀN!
     (segment D E)
     (parallel (segment A B) (segment D E))

3. ĐOẠN/ĐƯỜNG
   • (segment A B) - đoạn thẳng
   • (line A B) - đường thẳng

4. ĐƯỜNG TRÒN
   Khai báo:
   • (circle O) hoặc (circle O (radius 0.5))
   • (incircle A B C) / (circumcircle A B C)
   • LUÔN: (define O point) TRƯỚC → (circle O) SAU
   • Scale: **1 cm = 0.1 đơn vị** (5cm → 0.5, 10cm → 1.0)
   
   ĐƯỜNG KÍNH vs DÂY - CỰC KỲ QUAN TRỌNG:
   
   **ĐƯỜNG KÍNH** (đi qua tâm O):
   • "đường kính AB" / "đường kính MN" / "có đường kính AB" → BẮT BUỘC 4 thành phần:
     1. (segment A B)
     2. (on-circle A O)
     3. (on-circle B O)
     4. (on-segment O A B) ← TÂM O NẰM GIỮA - TUYỆT ĐỐI KHÔNG THIẾU!
   
   **DÂY THƯỜNG** (không qua tâm):
   • "dây AB" / "dây CD" / "lấy dây CD" → CHỈ 3 thành phần:
     1. (define C point) + (define D point) ← Định nghĩa điểm
     2. (on-circle C O) + (on-circle D O) ← BẮT BUỘC CẢ 2 ĐIỂM on-circle!
     3. (segment C D)
   
   ⚠️ LỖI PHỔ BIẾN VỀ DÂY:
   • SAI: (define C point) + (define D point) + (segment C D) - THIẾU on-circle!
   • SAI: (segment C D) + (on-circle C O) + (on-circle D O) - THIẾU define!
   • ĐÚNG: (define C point) + (on-circle C O) + (define D point) + (on-circle D O) + (segment C D)
   
   VÍ DỤ: "Lấy dây CD song song với AB"
   (define C point)
   (on-circle C O)     ← BẮT BUỘC!
   (define D point)
   (on-circle D O)     ← BẮT BUỘC!
   (segment C D)
   (parallel (segment A B) (segment C D))
   
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
   
   d) "Đường tròn (O) có đường kính MN. Vẽ tiếp tuyến tại M"
      (define O point)
      (circle O)
      (define M point)
      (on-circle M O)
      (define N point)
      (on-circle N O)
      (segment M N)
      (on-segment O M N)  ← Đường kính MN
      (define A point)    ← Điểm trên tiếp tuyến
      (define B point)    ← Điểm trên tiếp tuyến
      (segment A B)
      (on-segment M A B)  ← M nằm GIỮA A và B
      (tangent M (circle O) AB)  ← Tiếp tuyến tại M

5. RÀNG BUỘC
   • (parallel (segment B C) (segment D E))
   • (perpendicular (segment A B) (segment C D))
   • (tangent M (circle O) AB) - đường thẳng AB tiếp xúc với đường tròn (O) tại điểm M
   • (angle-equal A B C D E F) - ∠ABC = ∠DEF
   • (angle-measure B A C 60) - ∠BAC = 60° (đỉnh A ở GIỮA)
   • (on-segment M C D) - M nằm trên đoạn CD
   • (on-circle A O) - A trên đường tròn tâm O
   • (distance O A 0.5), (equal-distance O M O H)
   
   TIẾP TUYẾN - QUAN TRỌNG:
   
   ⚠️ QUY TẮC VÀNG: 
   • Tiếp điểm là ENDPOINT → KHÔNG dùng on-segment (phổ biến 90% trường hợp)
   • Mỗi điểm CHỈ define 1 LẦN
   • on-segment CẦN ĐÚNG 3 điểm khác nhau
   
   ĐỊNH NGHĨA TIẾP TUYẾN:
   • Tiếp tuyến là đường thẳng chỉ tiếp xúc với đường tròn tại DUY NHẤT MỘT điểm (gọi là tiếp điểm)
   • Tiếp tuyến KHÔNG CẮT đường tròn, chỉ chạm vào đường tròn tại tiếp điểm
   • Tiếp tuyến VUÔNG GÓC với bán kính của đường tròn tại chính tiếp điểm đó
   • Nếu M là tiếp điểm trên đường tròn (O), thì tiếp tuyến tại M ⊥ OM (bán kính)
   
   ĐỊNH NGHĨA TIẾP ĐIỂM:
   • Tiếp điểm là điểm DUY NHẤT mà tiếp tuyến và đường tròn tiếp xúc (chạm) nhau
   • Tiếp điểm PHẢI ĐỒNG THỜI:
     1. Nằm trên đường tròn
     2. Nằm trên đường thẳng tiếp tuyến
     3. Là điểm mà bán kính từ tâm đến tiếp điểm VUÔNG GÓC với tiếp tuyến
   • Ký hiệu: Nếu M là tiếp điểm của tiếp tuyến AB với đường tròn (O)
     → M ∈ (O) VÀ M ∈ AB VÀ OM ⊥ AB
   
   CÚ PHÁP: (tangent M (circle O) AB)
   • M = tiếp điểm (tangent point) - điểm duy nhất mà tiếp tuyến chạm đường tròn
   • (circle O) = đường tròn tâm O (nested structure)
   • AB = chuỗi 2 ký tự chỉ đường thẳng tiếp tuyến (ví dụ: "AB", "MN")
   
   ĐẶC ĐIỂM TIẾP ĐIỂM:
   • M phải nằm trên đường tròn: (on-circle M O) ← BẮT BUỘC!
   • M phải nằm trên đường thẳng tiếp tuyến - có 2 TRƯỜNG HỢP:
   
   TRƯỜNG HỢP 1: Tiếp điểm M LÀ ENDPOINT của segment (phổ biến nhất)
   • Ví dụ: "Từ A kẻ tiếp tuyến AM" - segment chỉ có 2 điểm A và M
   • M ĐÃ LÀ endpoint của segment → KHÔNG CẦN on-segment
   • Chỉ cần: (segment A M) + (tangent M (circle O) AM)
   
   TRƯỜNG HỢP 2: Tiếp điểm M NẰM GIỮA 2 endpoint của segment
   • Ví dụ: "Đường thẳng AB tiếp xúc tại M" - M nằm giữa A và B
   • M không phải endpoint → CẦN on-segment M A B
   • Cần: (segment A B) + (on-segment M A B) + (tangent M (circle O) AB)
   
   QUY TRÌNH VẼ TIẾP TUYẾN:
   1. Định nghĩa tiếp điểm M: (define M point)
   2. Đặt M trên đường tròn: (on-circle M O)
   3. Định nghĩa các điểm tạo tiếp tuyến: (define A point) [+ (define B point) nếu cần]
   4. ⚠️ NẾU đề nói "A nằm ngoài đường tròn": PHẢI thêm (distance O A <giá_trị>)
      • Với đường tròn bán kính R, chọn khoảng cách > R (ví dụ: R=0.5 → chọn 1.0 hoặc 1.5)
      • Nếu không có bán kính cụ thể, dùng 1.5 (giả sử bán kính mặc định ~1.0)
   5. Vẽ segment tiếp tuyến: (segment A M) hoặc (segment A B)
   6. CHỈ khi M NẰM GIỮA: (on-segment M A B)
   7. Khai báo tiếp tuyến: (tangent M (circle O) AM) hoặc (tangent M (circle O) AB)
   
   VÍ DỤ 1: "Đường tròn (O) bán kính 5cm, đường thẳng AB tiếp xúc với (O) tại M (M nằm giữa A và B)"
   (define O point)
   (circle O (radius 0.5))
   (define M point)           ← Tiếp điểm
   (on-circle M O)            ← M nằm trên đường tròn (O)
   (define A point)           ← Điểm A trên tiếp tuyến
   (define B point)           ← Điểm B trên tiếp tuyến
   (segment A B)              ← Vẽ segment AB
   (on-segment M A B)         ← M nằm GIỮA A và B - BẮT BUỘC!
   (tangent M (circle O) AB)  ← AB tiếp xúc (O) tại M
   
   VÍ DỤ 2: "Cho đường tròn (O). Từ điểm A ngoài đường tròn, kẻ tiếp tuyến AM đến (O) (M là tiếp điểm)"
   (define O point)
   (circle O)
   (define A point)           ← A ngoài đường tròn
   (distance O A 1.5)         ← ĐẢM BẢO A nằm ngoài (O) - BẮT BUỘC khi đề nói "ngoài"!
   (define M point)           ← M là tiếp điểm
   (on-circle M O)            ← M nằm trên (O) - BẮT BUỘC
   (segment A M)              ← AM là tiếp tuyến (M là endpoint)
   (tangent M (circle O) AM)  ← AM tiếp xúc (O) tại M - KHÔNG CẦN on-segment!
   
   VÍ DỤ 3: "Đường tròn (O; 6cm). Vẽ tiếp tuyến tại điểm T, tiếp tuyến cắt đường thẳng xy tại H"
   (define O point)
   (circle O (radius 0.6))
   (define T point)           ← T là tiếp điểm
   (on-circle T O)            ← T nằm trên (O)
   (define P point)           ← P trên tiếp tuyến
   (define Q point)           ← Q trên tiếp tuyến
   (segment P Q)              ← PQ là tiếp tuyến
   (on-segment T P Q)         ← T nằm GIỮA P và Q - BẮT BUỘC!
   (tangent T (circle O) PQ)  ← PQ tiếp xúc (O) tại T
   (define x point)
   (define y point)
   (segment x y)
   (define H point (inter-ll P Q x y))
   
   VÍ DỤ 4: "Từ điểm A ngoài (O) kẻ hai tiếp tuyến AB, AC. Gọi M là trung điểm BC. Chứng minh AM ⊥ BC"
   (define O point)
   (circle O)
   (define A point)
   (distance O A 1.5)         ← A ngoài (O)
   (define B point)           ← B là tiếp điểm thứ nhất
   (on-circle B O)
   (segment A B)
   (tangent B (circle O) AB)
   (define C point)           ← C là tiếp điểm thứ hai
   (on-circle C O)
   (segment A C)
   (tangent C (circle O) AC)
   (segment B C)              ← BẮT BUỘC trước midpoint!
   (define M point (midpoint B C))
   (segment A M)              ← Không constraint vuông góc vì đề yêu cầu CHỨNG MINH
   
   VÍ DỤ 5: "Đường tròn (O) có đường kính MN. Vẽ tiếp tuyến tại M. Chứng minh tiếp tuyến vuông góc MN"
   (define O point)
   (circle O)
   (define M point)           ← M chỉ define MỘT LẦN!
   (on-circle M O)
   (define N point)
   (on-circle N O)
   (segment M N)
   (on-segment O M N)         ← Đường kính MN - tâm O nằm giữa
   (define A point)           ← Điểm trên tiếp tuyến
   (define B point)           ← Điểm trên tiếp tuyến
   (segment A B)
   (on-segment M A B)         ← M nằm GIỮA A và B trên tiếp tuyến
   (tangent M (circle O) AB)  ← Tiếp tuyến AB tại M
   
   ⚠️ CÁC LỖI SAI PHỔ BIẾN VỀ TIẾP TUYẾN - TUYỆT ĐỐI TRÁNH:
   
   LỖI 1a: "AB là tiếp tuyến tại A" → THÊM (on-segment A B) - SAI 100%! ⚠️⚠️⚠️
   • Đề bài: "AB là tiếp tuyến tại A" / "AB tiếp xúc (O) tại A"
   • A là tiếp điểm VÀ A là endpoint của AB → TUYỆT ĐỐI KHÔNG DÙNG on-segment
   
   ❌ SAI (LỖI CỰC KỲ PHỔ BIẾN):
   (define A point)
   (on-circle A O)
   (define B point)
   (segment A B)
   (on-segment A B)           ← SAI VÌ CHỈ CÓ 2 ĐIỂM!
   (tangent A (circle O) AB)
   
   ✅ ĐÚNG:
   (define A point)
   (on-circle A O)
   (define B point)
   (segment A B)              ← A và B là 2 endpoint
   (tangent A (circle O) AB)  ← A là tiếp điểm - KHÔNG CẦN on-segment!
   
   LỖI 1b: (on-segment M A M) - SAI HOÀN TOÀN!
   • Không thể có 3 điểm mà 2 điểm giống nhau (M A M)
   • Nếu M là endpoint của segment AM → KHÔNG CẦN on-segment
   • ĐÚNG: Chỉ cần (segment A M) + (tangent M (circle O) AM)
   
   LỖI 2: (on-segment A B) - THIẾU ĐIỂM THỨ 3! ⚠️ CỰC KỲ NGHIÊM TRỌNG
   • on-segment CẦN ĐÚNG 3 điểm: (on-segment <điểm-giữa> <endpoint1> <endpoint2>)
   • ❌ SAI: (on-segment A B) - chỉ có 2 điểm
   • ❌ SAI: (on-segment M A M) - điểm trùng nhau
   • ❌ SAI: (on-segment O A O) - điểm trùng nhau
   • ✅ ĐÚNG: (on-segment M A B) - M nằm giữa A và B
   • ✅ ĐÚNG: (on-segment O M N) - O nằm giữa M và N (đường kính)
   • ✅ ĐÚNG: (on-segment H A C) - H nằm giữa A và C
   
   VÍ DỤ CHI TIẾT:
   • "M nằm trên đoạn AB" → (on-segment M A B)
   • "Đường kính MN" → (on-segment O M N) - O là tâm nằm giữa M và N
   • "H là hình chiếu O lên AC" → DÙNG (projection O (segment A C)) - TỰ ĐỘNG on-segment
   
   ⚠️ KHI NÀO KHÔNG CẦN on-segment?
   • Khi điểm là ENDPOINT của segment → KHÔNG CẦN
   • Ví dụ: (segment A M) - M là đầu mút → không cần (on-segment M A M)
   
   LỖI 3: Define điểm trùng lặp ⚠️ CỰC KỲ NGHIÊM TRỌNG
   • ❌ SAI: (define M point) ... (define M point) - M bị định nghĩa 2 lần
   • ✅ MỖI ĐIỂM CHỈ DEFINE 1 LẦN DUY NHẤT trong toàn bộ DSL
   
   VÍ DỤ SAI THƯỜNG GẶP: "Đường kính MN, tiếp tuyến AB tại M"
   ❌ SAI:
   (define M point)   ← Lần 1
   (on-circle M O)
   (define N point)
   (on-circle N O)
   (segment M N)
   (on-segment O M N)
   (define M point)   ← LẦN 2 - SAI!!! M đã được define ở trên rồi
   (define A point)
   ...
   
   ✅ ĐÚNG:
   (define M point)   ← CHỈ define M MỘT LẦN
   (on-circle M O)
   (define N point)
   (on-circle N O)
   (segment M N)
   (on-segment O M N)
   (define A point)   ← Define A - KHÔNG define M lại
   (define B point)
   (segment A B)
   (on-segment M A B)
   (tangent M (circle O) AB)
   
   ⚠️ CÁCH PHÁT HIỆN:
   • Trước khi viết (define X point), search toàn bộ DSL đã có X chưa
   • Nếu X đã tồn tại → CHỈ SỬ DỤNG, KHÔNG define lại
   • Thường gặp với tiếp điểm M, N, P khi có nhiều tiếp tuyến
   
   LỖI 4: Thiếu on-circle cho tiếp điểm
   • SAI: (define M point) + (tangent M (circle O) AM) - THIẾU on-circle
   • Tiếp điểm PHẢI nằm trên đường tròn
   • ĐÚNG: (define M point) + (on-circle M O) + (tangent M (circle O) AM)
   
   LỖI 5: Thiếu distance khi đề nói "điểm nằm ngoài đường tròn"
   • SAI: "Từ A nằm ngoài (O)" → chỉ có (define A point) - A có thể trùng O!
   • Không có constraint → optimizer có thể đặt A bất kỳ đâu, kể cả trùng tâm O
   • ĐÚNG: (define A point) + (distance O A 1.5) - đảm bảo A xa O hơn bán kính
   • Nguyên tắc: Với bán kính R, chọn khoảng cách > R (ví dụ: R=0.5 → dùng 1.0 hoặc 1.5)
   
   LỖI 6: Thêm constraint khi đề bài yêu cầu CHỨNG MINH ⚠️ CỰC KỲ NGHIÊM TRỌNG
   • ❌ SAI: "Chứng minh AB = AC" → KHÔNG ĐƯỢC thêm (equal-distance A B A C)
   • ❌ SAI: "Chứng minh AB ⟂ CD" → KHÔNG ĐƯỢC thêm (perpendicular ...)
   • ❌ SAI: "Chứng minh góc AOC = góc BAC" → KHÔNG ĐƯỢC thêm (angle-equal ...)
   • ❌ SAI: "Tính góc A" → KHÔNG ĐƯỢC thêm (angle-measure ...)
   • ❌ SAI: "Tìm độ dài AB" → KHÔNG ĐƯỢC thêm (distance-formula ...)
   • ✅ CHỈ thêm constraint khi đề bài CHO SẴN, không phải yêu cầu chứng minh
   
   VÍ DỤ CHI TIẾT:
   ❌ SAI: "Chứng minh rằng góc AOC = góc BAC"
   (angle-equal A O C B A C)  ← SAI - đây là điều cần chứng minh!
   
   ✅ ĐÚNG: Chỉ vẽ hình, KHÔNG thêm (angle-equal ...)
   
   ✅ ĐÚNG: "Cho góc A = 60°" → ĐƯỢC phép thêm
   (angle-measure B A C 60)  ← ĐÚNG - đề bài CHO SẴN
   
   ⚠️ PHÂN BIỆT:
   • "Cho ..." / "Biết ..." / "Với ..." → CHO SẴN → THÊM constraint
   • "Chứng minh ..." / "Tính ..." / "Tìm ..." → YÊU CẦU → KHÔNG thêm constraint
   
   LỖI 7: Thiếu segment trước khi dùng midpoint/projection
   • SAI: (define M point (midpoint B C)) - khi B C CHƯA ĐƯỢC NỐI
   • SAI: (define H point (projection A (segment B C))) - khi B C CHƯA ĐƯỢC NỐI
   • Optimizer cần segment B C phải tồn tại trước khi tính midpoint/projection
   • ĐÚNG (nếu chưa có BC): (segment B C) + (define M point (midpoint B C))
   • ĐÚNG (nếu đã có BC từ triangle): (triangle (A B C)) + (define M point (midpoint B C))
   
   LỖI 8: Thiếu on-circle cho dây cung
   • SAI: "Lấy dây CD song song AB" → (define C point) + (define D point) + (segment C D) - THIẾU on-circle!
   • Dây cung → CẢ 2 đầu đều PHẢI nằm trên đường tròn
   • ĐÚNG: (define C point) + (on-circle C O) + (define D point) + (on-circle D O) + (segment C D)
   
   LỖI 9: Sai cách dùng projection khi có perpendicular
   • SAI: "Kẻ OH ⊥ AC (H ∈ AC)" → (define H point) + (perpendicular (segment O H) (segment A C)) + (on-segment H A C)
   • ĐÚNG: Dùng (projection O (segment A C)) - tự động vuông góc và nằm trên AC
   
   GÓC Ở TÂM ĐƯỜNG TRÒN - CỰC KỲ QUAN TRỌNG:
   
   ĐỊNH NGHĨA: Góc có đỉnh trùng với tâm đường tròn được gọi là GÓC Ở TÂM.
   
   QUY TẮC VẼ GÓC Ở TÂM:
   1. Kiểm tra xem các điểm đã được nối với tâm O chưa
   2. CHỈ vẽ bán kính CHƯA CÓ (tránh trùng lặp)
   3. Kê khai góc: (angle-measure A O C 50)
   
   ⚠️ LƯU Ý ĐẶC BIỆT - ĐƯỜNG KÍNH:
   • Nếu AB là ĐƯỜNG KÍNH [(segment A B) + (on-segment O A B)]
     → O ĐÃ NỐI với A và B rồi
     → KHÔNG vẽ lại (segment O A) hay (segment O B) - bị TRÙNG!
   • CHỈ vẽ bán kính MỚI đến điểm chưa nối
   
   VÍ DỤ: "AB là đường kính, C trên đường tròn, góc AOC = 50°"
   (segment A B)        ← Đường kính
   (on-segment O A B)   ← O nằm giữa → O đã nối A và B
   (define C point)
   (on-circle C O)
   (segment O C)        ← CHỈ vẽ bán kính mới OC, KHÔNG vẽ OA (đã có)!
   (angle-measure A O C 50)
   
   VÍ DỤ: "Dây AB (không phải đường kính), góc AOB = 120°"
   (segment A B)        ← Dây thường (không có on-segment)
   (segment O A)        ← CẦN vẽ bán kính
   (segment O B)        ← CẦN vẽ bán kính  
   (angle-measure A O B 120)
   
   NHẬN BIẾT: Nếu đỉnh góc (chữ ở giữa) là tên tâm đường tròn → GÓC Ở TÂM → CẦN BÁN KÍNH!

═══ QUY TẮC ═══

1. THỨ TỰ: Hình → Define points → Segments/Lines → Circles → Constraints

2. GÓC: "góc A = 60°" → (angle-measure B A C 60) [A ở GIỮA]
   
   GÓC Ở TÂM: Nếu đỉnh góc là tâm đường tròn (góc có đỉnh trùng với tâm)
   → PHẢI vẽ bán kính đến các điểm CHƯA NỐI với tâm
   • Nếu AB là ĐƯỜNG KÍNH: O đã nối A, B → CHỈ vẽ bán kính mới
   • Nếu A, B là điểm tự do: CẦN vẽ cả 2 bán kính O-A và O-B
   
   VÍ DỤ 1: AB đường kính, góc AOC = 50°
   → (segment O C) + (angle-measure A O C 50)  [OA đã có từ đường kính]
   
   VÍ DỤ 2: Dây AB, góc AOB = 120°
   → (segment O A) + (segment O B) + (angle-measure A O B 120)

3. ĐƯỜNG ĐẶC BIỆT:
   • Đường cao AH: (define H point (projection A (segment B C))) + (segment A H)
   • Trung tuyến AM: (define M point (midpoint B C)) + (segment A M)
   • Phân giác AD: (define D point (bisector B A C)) + (segment A D)
   
   ⚠️ QUY TẮC MIDPOINT/PROJECTION:
   • TRƯỚC KHI dùng (midpoint B C) hoặc (projection A (segment B C))
   • Segment B C phải ĐÃ TỒN TẠI (từ triangle, quadrilateral, hoặc định nghĩa riêng)
   • Nếu B C CHƯA được nối → thêm (segment B C) trước midpoint/projection
   • Nếu B C ĐÃ CÓ SẴN (ví dụ từ triangle/quad) → KHÔNG thêm nữa (sẽ bị trùng!)
   • Ví dụ: (triangle (A B C)) → 3 cạnh AB, BC, CA đã có → (define M point (midpoint B C)) OK luôn
   • Ví dụ: B, C riêng lẻ → (segment B C) + (define M point (midpoint B C))

═══ VÍ DỤ ═══

1. "Tam giác ABC, M là trung điểm BC"
   (triangle (A B C))          ← Triangle tự động tạo 3 cạnh AB, BC, CA
   (define M point (midpoint B C))  ← BC đã có, không cần thêm segment!
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

5. "Trên đường tròn (O; R), lấy bốn điểm A, B, M, N sao cho AB đi qua O và MN không đi qua O"
   (define O point)
   (circle O)
   (define A point)
   (define B point)
   (segment A B)
   (on-circle A O)
   (on-circle B O)
   (on-segment O A B)  ← AB đi qua O (đường kính)
   (define M point)
   (define N point)
   (segment M N)
   (on-circle M O)
   (on-circle N O)  ← MN không qua O (dây thường, KHÔNG có on-segment)

6. "Cho đường tròn (I) có các dây cung AB, CD, EF. Biết rằng AB và CD đi qua tâm I, còn EF không đi qua I"
   (define I point)
   (circle I)
   (define A point)
   (define B point)
   (segment A B)
   (on-circle A I)
   (on-circle B I)
   (on-segment I A B)  ← AB đi qua I (đường kính)
   (define C point)
   (define D point)
   (segment C D)
   (on-circle C I)
   (on-circle D I)
   (on-segment I C D)  ← CD đi qua I (đường kính)
   (define E point)
   (define F point)
   (segment E F)
   (on-circle E I)
   (on-circle F I)  ← EF không qua I (dây thường)

7. "Cho đường tròn (O) bán kính 5 cm và bốn điểm A, B, C, D sao cho OA = 3 cm, OB = 4 cm, OC = 7 cm, OD = 5 cm"
   (define O point)
   (circle O (radius 0.5))
   (define A point)
   (segment O A)
   (distance O A 0.3)
   (define B point)
   (segment O B)
   (distance O B 0.4)
   (define C point)
   (segment O C)
   (distance O C 0.7)
   (define D point)
   (segment O D)
   (distance O D 0.5)

8. "Cho đường tròn (O) có đường kính AB. Lấy điểm C nằm trên (O) sao cho ∠AOC = 50°. Vẽ dây CD vuông góc với AB (D ∈ (O))"
   (define O point)
   (circle O)
   (define A point)
   (define B point)
   (segment A B)
   (on-circle A O)
   (on-circle B O)
   (on-segment O A B)  ← AB là đường kính → O đã nối với A và B!
   (define C point)
   (on-circle C O)  ← C nằm trên (O)
   (segment O C)  ← CHỈ vẽ bán kính OC (bán kính mới), KHÔNG vẽ OA (đã có từ đường kính AB)!
   (angle-measure A O C 50)
   (define D point)
   (on-circle D O)  ← D ∈ (O) - BẮT BUỘC phải có!
   (segment C D)
   (perpendicular (segment A B) (segment C D))

9. "Cho đường tròn (O) có đường kính AB. Vẽ dây CD vuông góc với AB (D ∈ (O)) và dây DE song song với AB (E ∈ (O))"
   (define O point)
   (circle O)
   (define A point)
   (define B point)
   (segment A B)
   (on-circle A O)
   (on-circle B O)
   (on-segment O A B)  ← AB là đường kính
   (define C point)
   (on-circle C O)  ← C ∈ (O)
   (define D point)
   (on-circle D O)  ← D ∈ (O) - BẮT BUỘC!
   (segment C D)
   (perpendicular (segment A B) (segment C D))
   (define E point)
   (on-circle E O)  ← E ∈ (O) - BẮT BUỘC!
   (segment D E)
   (parallel (segment A B) (segment D E))

10. "Cho đường tròn (O) bán kính 6cm. Đường thẳng AB tiếp xúc với (O) tại điểm M (M nằm giữa A và B)"
   (define O point)
   (circle O (radius 0.6))
   (define M point)
   (on-circle M O)
   (define A point)
   (define B point)
   (segment A B)
   (on-segment M A B)  ← M nằm GIỮA A và B - CẦN on-segment
   (tangent M (circle O) AB)

10b. ⚠️ "Cho đường tròn (O). AB là tiếp tuyến tại A" (A LÀ ENDPOINT - KHÔNG CẦN on-segment)
   (define O point)
   (circle O)
   (define A point)
   (on-circle A O)          ← A là tiếp điểm
   (define B point)
   (segment A B)            ← A và B là 2 endpoint - KHÔNG CẦN on-segment!
   (tangent A (circle O) AB)
   
   ⚠️ SAI: (on-segment A B) - vì chỉ có 2 điểm!
   ⚠️ SAI: (on-segment A A B) - vì điểm trùng!

10c. ⚠️ "Cho đường tròn (O). Từ A ngoài (O) kẻ tiếp tuyến AM" (M LÀ ENDPOINT - KHÔNG CẦN on-segment)
   (define O point)
   (circle O)
   (define A point)
   (distance O A 1.5)       ← A nằm ngoài
   (define M point)
   (on-circle M O)          ← M là tiếp điểm
   (segment A M)            ← A và M là 2 endpoint - KHÔNG CẦN on-segment!
   (tangent M (circle O) AM)
   
   ⚠️ SAI: (on-segment M A M) - điểm trùng!
   ⚠️ SAI: (on-segment A M) - thiếu điểm thứ 3!


11. "Cho đường tròn (O) bán kính 5cm. Từ điểm A ngoài đường tròn, kẻ hai tiếp tuyến AM và AN đến (O) (M, N là các tiếp điểm)"
   (define O point)
   (circle O (radius 0.5))
   (define A point)           ← Điểm ngoài đường tròn
   (distance O A 1.0)         ← ĐẢM BẢO A nằm ngoài (radius=0.5 → chọn 1.0 > 0.5)
   (define M point)           ← Tiếp điểm thứ nhất
   (on-circle M O)            ← M nằm trên (O) - BẮT BUỘC
   (segment A M)              ← Vẽ tiếp tuyến AM (M là endpoint)
   (tangent M (circle O) AM)  ← AM tiếp xúc (O) tại M - KHÔNG CẦN on-segment
   (define N point)           ← Tiếp điểm thứ hai
   (on-circle N O)            ← N nằm trên (O) - BẮT BUỘC
   (segment A N)              ← Vẽ tiếp tuyến AN (N là endpoint)
   (tangent N (circle O) AN)  ← AN tiếp xúc (O) tại N - KHÔNG CẦN on-segment
   
   ⚠️ LƯU Ý VỀ NHIỀU TIẾP TUYẾN:
   • Mỗi tiếp tuyến cần có TIẾP ĐIỂM RIÊNG (M, N, P,...)
   • Mỗi tiếp tuyến cần KHAI BÁO RIÊNG: (tangent ...), (tangent ...)
   • Hệ thống HỖ TRỢ vẽ NHIỀU tiếp tuyến cùng lúc, không giới hạn số lượng

═══ MẪU DSL CHO CÁC DẠNG BÀI TIẾP TUYẾN PHỔ BIẾN ═══

DẠNG 1: "AB là tiếp tuyến tại A" - A VỪA LÀ TIẾP ĐIỂM VỪA LÀ ENDPOINT
(define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(segment A B)
(tangent A (circle O) AB)
⚠️ KHÔNG DÙNG (on-segment A B) - chỉ có 2 điểm!

DẠNG 2: "Từ A ngoài (O) kẻ tiếp tuyến AB và AC (B, C tiếp điểm)" - HAI TIẾP TUYẾN
(define O point)
(circle O)
(define A point)
(distance O A 1.5)          ← A nằm ngoài - BẮT BUỘC
(define B point)
(on-circle B O)
(segment A B)
(tangent B (circle O) AB)   ← Tiếp tuyến thứ nhất
(define C point)
(on-circle C O)
(segment A C)
(tangent C (circle O) AC)   ← Tiếp tuyến thứ hai
⚠️ KHÔNG DÙNG (on-segment ...) - B và C đều là endpoint!

DẠNG 3: "Đường kính MN, tiếp tuyến AB tại M" - TIẾP ĐIỂM TRÙNG ĐIỂM ĐƯỜNG KÍNH
(define O point)
(circle O)
(define M point)            ← CHỈ define M MỘT LẦN!
(on-circle M O)
(define N point)
(on-circle N O)
(segment M N)
(on-segment O M N)          ← Đường kính
(define A point)            ← KHÔNG define M lại!
(define B point)
(segment A B)
(on-segment M A B)          ← M nằm GIỮA A và B → CẦN on-segment
(tangent M (circle O) AB)
⚠️ KHÔNG define M 2 lần!
⚠️ M nằm giữa A và B → CẦN (on-segment M A B)

DẠNG 4: "AB tiếp xúc tại M (M nằm giữa A và B)" - M NẰM GIỮA 2 ENDPOINT
(define O point)
(circle O)
(define M point)
(on-circle M O)
(define A point)
(define B point)
(segment A B)
(on-segment M A B)          ← M nằm GIỮA - CẦN on-segment
(tangent M (circle O) AB)

DẠNG 5: "AB là tiếp tuyến, AC là dây" - KẾT HỢP TIẾP TUYẾN VÀ DÂY
(define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(segment A B)
(tangent A (circle O) AB)   ← A là endpoint - KHÔNG CẦN on-segment
(define C point)
(on-circle C O)             ← C là đầu kia của dây - BẮT BUỘC on-circle
(segment A C)

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

⚠️ CHECKLIST TRƯỚC KHI OUTPUT - KIỂM TRA KỸ DSL:

═══ KIỂM TRA BẮT BUỘC - TUYỆT ĐỐI KHÔNG ĐƯỢC BỎ QUA ═══

1. ✓ Mỗi điểm chỉ được (define ... point) MỘT LẦN duy nhất
   → Tìm toàn bộ DSL, đếm số lần define cho mỗi điểm - PHẢI = 1

2. ✓ Mọi on-segment đều có ĐÚNG 3 điểm: (on-segment M A B)
   → Không được có (on-segment A B) - thiếu điểm giữa
   → Không được có (on-segment M A M) - 2 điểm giống nhau

3. ✓ Tiếp điểm PHẢI có (on-circle M O) trước (tangent ...)
   → Thứ tự: (define M point) → (on-circle M O) → (tangent ...)

4. ✓ Tiếp điểm là endpoint → KHÔNG CẦN on-segment
   → (segment A M) + (tangent M (circle O) AM) - M là đầu mút
   → Tiếp điểm nằm giữa → (on-segment M A B) + (tangent M (circle O) AB)

5. ✓ Đề nói "nằm ngoài đường tròn" → PHẢI có (distance O A <value>)
   → Value > bán kính (radius 0.5 → dùng 1.0 hoặc 1.5)

6. ✓ KHÔNG thêm constraint cho yêu cầu CHỨNG MINH
   → "Chứng minh ..." / "Tính ..." / "Tìm ..." → KHÔNG thêm constraint
   → "Cho ..." / "Biết ..." → ĐƯỢC thêm constraint

7. ✓ Điểm trên đường tròn đều có (on-circle ... O)
   → Tìm tất cả điểm nằm trên đường tròn trong đề bài
   → Kiểm tra từng điểm có (on-circle ...) chưa

8. ✓ (midpoint B C) hoặc (projection A (segment B C)) → segment B C phải ĐÃ TỒN TẠI
   → Có (segment B C) hoặc (triangle (... B C)) trước đó

9. ✓ DÂY CUNG: "lấy dây CD" → CẢ C và D đều PHẢI có (on-circle C O) + (on-circle D O)
   → Kiểm tra: (define C point)? (on-circle C O)? (define D point)? (on-circle D O)? ✓

10. ✓ ĐƯỜNG KÍNH: "đường kính MN" → PHẢI có (on-segment O M N) để tâm nằm giữa
    → Đường kính = segment qua tâm: (segment M N) + (on-circle M O) + (on-circle N O) + (on-segment O M N)

11. ✓ H ∈ AC và OH ⊥ AC → dùng (projection O (segment A C)), KHÔNG define H riêng + perpendicular
    → SAI: (define H point) + (perpendicular ...) + (on-segment H A C)
    → ĐÚNG: (define H point (projection O (segment A C)))

12. ✓ Tiếp tuyến: Kiểm tra format (tangent <tiếp-điểm> (circle <tâm>) <segment>)
    → (tangent M (circle O) AB) - M là tiếp điểm, O là tâm, AB là segment

13. ✓ Không có comment # trong DSL output
    → Chỉ có DSL thuần túy với \n

14. ✓ Field "instruction" là bài toán gốc tiếng Việt, KHÔNG được thay đổi
    → Copy nguyên văn đề bài, không dịch sang tiếng Anh

15. ✓ Kiểm tra lại lần cuối: Có điểm nào được define 2 lần không?
    → Ctrl+F search "(define A point)", "(define B point)", ... trong DSL của bạn

═══ BƯỚC CUỐI CÙNG - XÁC MINH DSL ═══

⚠️ TRƯỚC KHI OUTPUT, PHÂN TÍCH ĐỀ BÀI VỀ TIẾP TUYẾN:

Bước 1: Tìm từ khóa tiếp tuyến trong đề
• "AB là tiếp tuyến tại A"
• "Từ A kẻ tiếp tuyến AM"
• "AB tiếp xúc (O) tại M"
• "Tiếp tuyến tại M"

Bước 2: Xác định tiếp điểm và vị trí
• Nếu: "tại A" hoặc "tại M" → tiếp điểm là A hoặc M
• Nếu: "AM" và M là tiếp điểm → M là endpoint của AM
• Nếu: "AB" và A là tiếp điểm → A là endpoint của AB
• Nếu: "M nằm giữa A và B" → M không phải endpoint

Bước 3: Quyết định dùng on-segment hay không
• Tiếp điểm là endpoint (A trong AB, M trong AM) → KHÔNG DÙNG on-segment
• Tiếp điểm nằm giữa (M giữa A và B) → DÙNG (on-segment M A B)

Bước 4: Kiểm tra DSL
• Tìm (on-segment A B) - chỉ 2 điểm → XÓA ngay!
• Tìm (on-segment M A M) - điểm trùng → XÓA ngay!
• Đếm define cho mỗi điểm → nếu > 1 → XÓA các lần define thừa

⚠️ TRƯỚC KHI OUTPUT, KIỂM TRA TỪNG DÒNG DSL:

1. Đếm số lần define cho MỖI điểm → Phải = 1
   • Tìm: (define A point) - có bao nhiêu lần?
   • Tìm: (define M point) - có bao nhiêu lần?
   • Nếu > 1 → XÓA các lần define thừa

2. Tìm tất cả (on-segment ...) → Đếm số điểm
   • (on-segment M A B) - 3 điểm ✓
   • (on-segment A B) - 2 điểm ✗ → XÓA (A là endpoint, không cần on-segment)
   • (on-segment M A M) - điểm trùng ✗ → XÓA
   • (on-segment A A B) - điểm trùng ✗ → XÓA
   
   ⚠️ QUY TẮC ĐẶC BIỆT CHO TIẾP TUYẾN:
   • "AB là tiếp tuyến tại A" → A là endpoint → KHÔNG DÙNG on-segment
   • "Từ A kẻ tiếp tuyến AM" → M là endpoint → KHÔNG DÙNG on-segment
   • "AB tiếp xúc tại M (M giữa A và B)" → M nằm giữa → CẦN (on-segment M A B)

3. Tìm tất cả (tangent ...) → Kiểm tra có (on-circle ...) TRƯỚC không?
   • (tangent M (circle O) AB) → Phải có (on-circle M O) ở trên
   • Nếu thiếu → THÊM (on-circle M O) ngay sau (define M point)

4. Tìm "dây" trong đề bài → Kiểm tra có (on-circle ...) cho CẢ 2 đầu không?
   • "Dây CD" → Cần (on-circle C O) VÀ (on-circle D O)
   • Thiếu 1 trong 2 → THÊM vào

5. Tìm yêu cầu "Chứng minh" / "Tính" / "Tìm" → Kiểm tra có thêm constraint không?
   • "Chứng minh AB = AC" → KHÔNG được có (equal-distance ...)
   • "Chứng minh góc AOC = góc BAC" → KHÔNG được có (angle-equal ...)
   • Nếu có → XÓA constraint đó đi

═══ INPUT ═══
{{ extract }}

"""
