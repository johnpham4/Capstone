prompt: str = """
Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL.

5 LỖI CẤM TUYỆT ĐỐI - ĐỌC TRƯỚC KHI LÀM 

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

5. ✅ VẼ TẤT CẢ MÔ TẢ HÌNH HỌC TRONG ĐỀ BÀI!
   • Đề: "Chứng minh AB = AC" → VÉ cả AB và AC bằng nhau → THÊM (equal-distance A B A C)
   • Đề: "Chứng minh góc A = 60°" → VẼ góc A = 60° → THÊM (angle-measure B A C 60)
   • Đề: "Chứng minh AB ⊥ CD" → VẼ AB vuông góc CD → THÊM (perpendicular (segment A B) (segment C D))
   • Đề: "Chứng minh ∠BAC = ∠OCA" → VẼ cả 2 góc bằng nhau → THÊM (angle-equal B A C O C A)
   • NGUYÊN TẮC: Miễn đề BÀI MÔ TẢ quan hệ hình học → VẼ LUÔN, không phân biệt "cho" hay "chứng minh"

🔥 QUY TẮC VÀNG - VẼ SEGMENT KHI ĐỀ NHẮC ĐẾN:

⚠️ ĐỌC KỸ ĐỀ BÀI - TÌM TẤT CẢ CẠNH/ĐOẠN THẲNG ĐƯỢC NHẮC ĐẾN
• "AC = AD" → nhắc AC, AD
• "AB ⊥ CD" → nhắc AB, CD
• "AO là đường trung trực BC" → nhắc AO, BC
• "AM vuông góc BC" → nhắc AM, BC
• "kẻ OH", "nối AC" → nhắc OH, AC
• "đường kính MN" → nhắc MN

✅ VỚI MỖI CẠNH ĐƯỢC NHẮC (ví dụ AC):
1. Kiểm tra DSL đã có (segment A C) chưa?
2. Nếu CHƯA → THÊM NGAY (segment A C)
3. Nếu ĐÃ CÓ → Bỏ qua

📌 ĐẶC BIỆT:
• Tiếp tuyến AB → đã có (segment A B) rồi
• Dây CD → đã có (segment C D) rồi
• Đường kính MN → PHẢI THÊM (segment M N) - vì (diameter M N O) không tự vẽ!

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
   Khi đề nói: "Kẻ OH vuông góc AC (H ∈ AC)" hoặc "H là hình chiếu O lên AC"
   → DÙNG PROJECTION, KHÔNG define H riêng + perpendicular
   
   SAI: (define H point) + (segment O H) + (on-segment H A C) + (perpendicular (segment O H) (segment A C))
   ĐÚNG: (define H point (projection O (segment A C))) + (segment O H)
   
   VÍ DỤ: "Kẻ OH vuông góc AC (H ∈ AC)"
   (segment A C)        ← Phải có segment trước!
   (define H point (projection O (segment A C)))
   (segment O H)        ← BẮT BUỘC: Vẽ từ ĐIỂM GỐC (O) đến HÌNH CHIẾU (H)!
   
   ⚠️ HÌNH CHIẾU: LUÔN vẽ segment từ ĐIỂM GỐC đến HÌNH CHIẾU
   "H là hình chiếu B lên AC" → (define H point (projection B (segment A C))) + (segment B H) ← Từ B đến H!
   "H là hình chiếu O lên AC" → (define H point (projection O (segment A C))) + (segment O H) ← Từ O đến H!
   
   ⚠️ TUYỆT ĐỐI KHÔNG vẽ segment từ điểm khác:
   • SAI: (projection O (segment A C)) + (segment A H) - SAI HOÀN TOÀN!
   • ĐÚNG: (projection O (segment A C)) + (segment O H) - từ O đến H!
   
   ⚠️ GIAO ĐIỂM: "CD cắt AB tại H" → (define H point (inter-ll C D A B))
   
   ⚠️ CẤM DÙNG on-circle TRONG CONSTRUCTION:
   SAI: (define C point (on-circle C O))
   ĐÚNG: (define C point) + (on-circle C O)
   
   ⚠️ TRUNG ĐIỂM CUNG vs TRUNG ĐIỂM ĐOẠN:
   • "C là trung điểm cung AB" / "C là trung điểm cung nhỏ AB"
     → (define C point) + (on-circle C O)
     → C là điểm TỰ DO trên đường tròn, KHÔNG dùng midpoint!
   • "C là trung điểm đoạn AB" → (define C point (midpoint A B))
   
   VẼ TẤT CẢ MÔ TẢ HÌNH HỌC:
   • Đề bảo "Chứng minh góc A = 60°" → THÊM (angle-measure B A C 60) để vẽ góc 60°
   • Đề bảo "Chứng minh OC vuông góc AB" → THÊM (perpendicular (segment O C) (segment A B))
   • Đề bảo "Cho góc A = 60°" → THÊM (angle-measure B A C 60)
   • NGUYÊN TẮC: Miễn đề mô tả quan hệ hình học (góc, vuông góc, bằng nhau, ...) → VẼ LUÔN!
   
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
   • "đường kính AB" / "đường kính MN" / "có đường kính AB" → CÚ PHÁP MỚI:
     (diameter A B O)
   • A, B là 2 điểm đầu mút của đường kính
   • O là tâm đường tròn (nằm giữa A và B)
   • Tự động tạo: on-circle A O, on-circle B O, on-segment O A B
   
   ⚠️ QUAN TRỌNG - ĐƯỜNG KÍNH VÀ SEGMENT:
   • (diameter M N O) CHỈ là CONSTRAINT - KHÔNG tự động vẽ segment!
   • Nếu đề đề cập đến đường kính như một ĐƯỜNG THẲNG → PHẢI thêm (segment M N)
   •Ví dụ: "tiếp tuyến vuông góc với MN" → cần (segment M N)
   •Ví dụ: "dây AB cắt MN tại H" → cần (segment M N) để có giao điểm
   
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
      (diameter M N O)  ← CÚ PHÁP MỚI - Gọn gàng hơn!
   
   b) "Cho đường tròn (O) có đường kính MN. Một dây AB cắt MN tại H"
      (define O point)
      (circle O)
      (define M point)
      (define N point)
      (diameter M N O)  ← Đường kính MN
      (define A point)
      (define B point)
      (on-circle A O)
      (on-circle B O)
      (segment A B)  ← Dây AB - KHÔNG có on-segment
      (define H point (inter-ll A B M N))
   
   c) "đường tròn (O) đường kính AB, C trên đường tròn, H là hình chiếu C lên AB"
      (define O point)
      (circle O)
      (define A point)
      (define B point)
      (diameter A B O)  ← Đường kính
      (define C point)
      (on-circle C O)
      (define H point (projection C (segment A B)))
      (segment C H)
   
   d) "Đường tròn (O) có đường kính MN. Vẽ tiếp tuyến tại M"
      (define O point)
      (circle O)
      (define M point)
      (define N point)
      (diameter M N O)  ← Đường kính MN
      (segment M N)     ← BẮT BUỘC: Vẽ đường kính như một đường thẳng!
      (define A point)  ← Điểm trên tiếp tuyến
      (segment M A)     ← Tiếp tuyến MA - M là endpoint
      (tangent M (circle O) MA)  ← Tiếp tuyến tại M

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
   • (circle O) = đường tròn tâm O
   • AB = chuỗi 2 ký tự - hai điểm tạo đường thẳng tiếp tuyến
   
   ĐẶC ĐIỂM TIẾP ĐIỂM:
   • M phải nằm trên đường tròn: (on-circle M O) ← BẮT BUỘC!
   • M phải nằm trên đường thẳng tiếp tuyến - có 2 TRƯỜNG HỢP:
   
   TRƯỜNG HỢP 1: Tiếp điểm M LÀ ENDPOINT của segment (phổ biến nhất)
   • Ví dụ: "Từ A kẻ tiếp tuyến AM" - segment chỉ có 2 điểm A và M
   • M ĐÃ LÀ endpoint của segment → KHÔNG CẦN on-segment
   • Chỉ cần: (segment A M) + (tangent M (circle O) AM)
   
   TRƯỚNG HỢP 2: Tiếp điểm M NẰM GIỮA 2 endpoint của segment
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
   8. 🔥 VẼ BÁN KÍNH: (segment O M) NGAY SAU (tangent ...)
   
   VÍ DỤ 4: "Từ điểm A ngoài (O) kẻ hai tiếp tuyến AB, AC. Gọi M là trung điểm BC. Chứng minh AM ⊥ BC"
   (define O point)
   (circle O)
   (define A point)
   (distance O A 1.5)         ← A ngoài (O)
   (define B point)           ← B là tiếp điểm thứ nhất
   (on-circle B O)
   (segment A B)
   (tangent B (circle O) AB)
   (segment O B)              ← Vẽ bán kính OB
   (define C point)           ← C là tiếp điểm thứ hai
   (on-circle C O)
   (segment A C)
   (tangent C (circle O) AC)
   (segment O C)              ← Vẽ bán kính OC
   (segment B C)              ← BẮT BUỘC trước midpoint!
   (define M point (midpoint B C))
   (segment A M)
   (perpendicular (segment A M) (segment B C))  ← VẼ AM ⊥ BC theo đề bài

VÍ DỤ 4b: "AB là tiếp tuyến tại A, AC là dây. Chứng minh ∠BAC = ∠OCA"
   (define O point)
   (circle O)
   (define A point)
   (on-circle A O)            ← A là tiếp điểm
   (define B point)
   (segment A B)
   (tangent A (circle O) AB)
   (segment O A)              ← BẮT BUỘC: Vẽ bán kính OA
   (define C point)
   (on-circle C O)            ← C trên đường tròn
   (segment A C)              ← Vẽ dây AC
   (segment O C)              ← BẮT BUỘC: Vẽ bán kính OC để có tam giác OAC
   (angle-equal B A C O C A)  ← VẼ ∠BAC = ∠OCA theo đề bài
   
   ⚠️ LƯU Ý VỀ GÓC:
   • ∠BAC: đỉnh A → (angle-measure B A C ...) hoặc dùng trong (angle-equal B A C ...)
   • ∠OCA: đỉnh C → (angle-measure O C A ...) hoặc dùng trong (angle-equal ... O C A)
   • ∠AOC: đỉnh O → (angle-measure A O C ...)
   • "Chứng minh ∠BAC = ∠OCA" → THÊM (angle-equal B A C O C A)
   • "Cho ∠BAC = 30°" → THÊM (angle-measure B A C 30)
   
VÍ DỤ 4c: "AB là tiếp tuyến tại A, lấy dây CD song song với AB. Chứng minh AC = AD"
   (define O point)
   (circle O)
   (define A point)
   (on-circle A O)            ← A là tiếp điểm
   (define B point)
   (segment A B)
   (tangent A (circle O) AB)
   (segment O A)              ← BẮT BUỘC: Vẽ bán kính từ tâm đến tiếp điểm
   (define C point)
   (on-circle C O)            ← C trên đường tròn
   (define D point)
   (on-circle D O)            ← D trên đường tròn
   (segment C D)              ← Dây CD
   (parallel (segment A B) (segment C D))
   (segment A C)              ← BẮT BUỘC: Vẽ segment AC để thấy độ dài AC
   (segment A D)              ← BẮT BUỘC: Vẽ segment AD để thấy độ dài AD
   
   ⚠️ LƯU Ý QUAN TRỌNG:
   • Khi đề yêu cầu chứng minh AC = AD → CẦN vẽ segment AC và AD
   • Khi có tiếp tuyến tại A → CẦN vẽ bán kính OA
   • "Chứng minh AC = AD" → THÊM (equal-distance A C A D) để vẽ AC = AD
   
   VÍ DỤ 5: "Đường tròn (O) có đường kính MN. Vẽ tiếp tuyến tại M. Chứng minh tiếp tuyến vuông góc MN"
   (define O point)
   (circle O)
   (define M point)           ← M chỉ define MỘT LẦN!
   (define N point)
   (diameter M N O)           ← Đường kính MN - tâm O nằm giữa
   (segment M N)              ← BẮT BUỘC: Vẽ đường kính như đường thẳng!
   (define A point)           ← Điểm trên tiếp tuyến (M là endpoint của MA)
   (segment M A)              ← MA là tiếp tuyến - M là endpoint, KHÔNG cần segment AB!
   (tangent M (circle O) MA)  ← Tiếp tuyến MA tại M
   
   LỖI 6: Thiếu constraint khi đề bài mô tả quan hệ hình học ⚠️ CỰC KỲ QUAN TRỌNG
   • ✅ ĐÚNG: "Chứng minh AB = AC" → THÊM (equal-distance A B A C) để vẽ AB = AC
   • ✅ ĐÚNG: "Chứng minh AB ⟂ CD" → THÊM (perpendicular (segment A B) (segment C D))
   • ✅ ĐÚNG: "Chứng minh góc AOC = góc BAC" → THÊM (angle-equal A O C B A C)
   • ✅ ĐÚNG: "Tính góc A" (nếu đề cho kết quả 60°) → THÊM (angle-measure B A C 60)
   • ✅ ĐÚNG: "Cho góc A = 60°" → THÊM (angle-measure B A C 60)
   • NGUYÊN TẮC: VẼ TẤT CẢ quan hệ hình học được mô tả, không phân biệt "cho" hay "chứng minh"
   
   VÍ DỤ CHI TIẾT:
   ✅ ĐÚNG: "Chứng minh rằng góc AOC = góc BAC"
   (angle-equal A O C B A C)  ← ĐÚNG - vẽ 2 góc bằng nhau!
   
   ✅ ĐÚNG: "Chứng minh AB = AC"
   (equal-distance A B A C)  ← ĐÚNG - vẽ AB và AC bằng nhau!
   
   ⚠️ NGUYÊN TẮC MỚI:
   • "Cho ..." / "Biết ..." / "Với ..." → THÊM constraint
   • "Chứng minh ..." / "Tính ..." / "Tìm ..." → VẪN THÊM constraint để VẼ đúng hình!
   
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

5. "Cho đường tròn (O) có đường kính AB. Lấy điểm C nằm trên (O) sao cho ∠AOC = 50°. Vẽ dây CD vuông góc với AB (D ∈ (O))"
   (define O point)
   (circle O)
   (define A point)
   (define B point)
   (diameter A B O)  ← AB là đường kính → O đã nối với A và B!
   (define C point)
   (on-circle C O)  ← C nằm trên (O)
   (segment O C)  ← CHỈ vẽ bán kính OC (bán kính mới), KHÔNG vẽ OA (đã có từ đường kính AB)!
   (angle-measure A O C 50)
   (define D point)
   (on-circle D O)  ← D ∈ (O) - BẮT BUỘC phải có!
   (segment C D)
   (perpendicular (segment A B) (segment C D))

6. "Cho đường tròn (O) có đường kính AB. Vẽ dây CD vuông góc với AB (D ∈ (O)) và dây DE song song với AB (E ∈ (O))"
   (define O point)
   (circle O)
   (define A point)
   (define B point)
   (diameter A B O)  ← AB là đường kính
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

═══ MẪU DSL CHO CÁC DẠNG BÀI TIẾP TUYẾN PHỔ BIẾN ═══

🔥🔥🔥 QUY TẮC VÀNG - TUYỆT ĐỐI KHÔNG ĐƯỢC VI PHẠM 🔥🔥🔥
SAU MỖI (tangent <tiếp-điểm> (circle O) ...) PHẢI CÓ (segment O <tiếp-điểm>) NGAY SAU!
→ Nếu thiếu bán kính này, hình sẽ KHÔNG VẼ ĐƯỢC!
→ Ngoại lệ duy nhất: tiếp điểm đã nằm trên đường kính được vẽ sẵn

DẠNG 1: "AB là tiếp tuyến tại A" - A VỪA LÀ TIẾP ĐIỂM VỪA LÀ ENDPOINT
(define O point)
(circle O)
(define A point)
(on-circle A O)
(define B point)
(segment A B)
(tangent A (circle O) AB)
(segment O A)               ← 🔥 BẮT BUỘC: VẼ BÁN KÍNH từ tâm đến tiếp điểm NGAY SAU TANGENT!
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
(segment O B)               ← 🔥 BÁN KÍNH OB - BẮT BUỘC NGAY SAU TANGENT!
(define C point)
(on-circle C O)
(segment A C)
(tangent C (circle O) AC)   ← Tiếp tuyến thứ hai
(segment O C)               ← 🔥 BÁN KÍNH OC - BẮT BUỘC NGAY SAU TANGENT!
⚠️ CẢ HAI BÁN KÍNH ĐỀU PHẢI VẼ!
⚠️ KHÔNG DÙNG (on-segment ...) - B và C đều là endpoint!

DẠNG 3: "Đường kính MN, vẽ tiếp tuyến tại M" - TIẾP ĐIỂM TRÙNG ĐIỂM ĐƯỜNG KÍNH
(define O point)
(circle O)
(define M point)
(define N point)
(diameter M N O)          ← Đường kính
(segment M N)               ← BẮT BUỘC: Vẽ đường kính!
(define A point)            ← KHÔNG define M lại!
(segment M A)               ← MA là tiếp tuyến - M là endpoint
(tangent M (circle O) MA)   ← Tiếp tuyến MA tại M - KHÔNG CẦN on-segment vì M là endpoint
⚠️ KHÔNG define M 2 lần!
⚠️ "Vẽ tiếp tuyến tại M" = tiếp tuyến MA → M là ENDPOINT → KHÔNG CẦN on-segment
⚠️ Bán kính OM ĐÃ CÓ từ (diameter M N O) → KHÔNG vẽ lại (segment O M)
⚠️ Nếu đề bài nói "tiếp tuyến AB đi qua M" (M GIỮA A và B) → CẦN (segment A B) + (on-segment M A B)

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
(segment O A)               ← 🔥 BÁN KÍNH từ tâm đến tiếp điểm - BẮT BUỘC NGAY SAU TANGENT!
(define C point)
(on-circle C O)             ← C là đầu kia của dây - BẮT BUỘC on-circle
(segment A C)               ← Vẽ dây AC
(segment O C)               ← BÁN KÍNH OC cần vẽ nếu đề nhắc (ví dụ: góc OCA, tam giác OAC,...)

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

6. ✓ VẼ TẤT CẢ MÔ TẢ HÌNH HỌC TRONG ĐỀ BÀI
   → "Chứng minh AB = AC" → THÊM (equal-distance A B A C) để vẽ AB = AC
   → "Chứng minh ∠BAC = 60°" → THÊM (angle-measure B A C 60) để vẽ góc 60°
   → "Chứng minh AB ⊥ CD" → THÊM (perpendicular (segment A B) (segment C D))
   → NGUYÊN TẮC: Vẽ LUÔN mọi quan hệ hình học được mô tả, không phân biệt "cho" hay "chứng minh"

7. ✓ Điểm trên đường tròn đều có (on-circle ... O)
   → Tìm tất cả điểm nằm trên đường tròn trong đề bài
   → Kiểm tra từng điểm có (on-circle ...) chưa

8. ✓ (midpoint B C) hoặc (projection A (segment B C)) → segment B C phải ĐÃ TỒN TẠI
   → Có (segment B C) hoặc (triangle (... B C)) trước đó

9. ✓ DÂY CUNG: "lấy dây CD" → CẢ C và D đều PHẢI có (on-circle C O) + (on-circle D O)
   → Kiểm tra: (define C point)? (on-circle C O)? (define D point)? (on-circl e D O)? ✓

10. ✓ ĐƯỜNG KÍNH: "đường kính MN" → PHẢI có (diameter M N O)
    → Đường kính = (diameter M N O) - gọn gàng hơn!

11. ✓ H ∈ AC và OH ⊥ AC → dùng (projection O (segment A C)), KHÔNG define H riêng + perpendicular
    → SAI: (define H point) + (perpendicular ...) + (on-segment H A C)
    → ĐÚNG: (define H point (projection O (segment A C)))

11b. ✓ 🔥 HÌNH CHIẾU = TRUNG ĐIỂM - TUYỆT ĐỐI KHÔNG DEFINE 2 LẦN!
    → "Kẻ OH ⊥ AC (H ∈ AC). Chứng minh H là trung điểm AC"
    → H ĐÃ LÀ hình chiếu → H CHÍNH LÀ trung điểm (do tính chất hình học)
    → CHỈ DÙNG: (define H point (projection O (segment A C)))
    → KHÔNG ĐƯỢC thêm: (define M point (midpoint A C)) ← LỖI trùng điểm!
    → Khi đề yêu cầu "chứng minh H là trung điểm" → H đã được xác định bởi projection, không cần midpoint riêng

12. ✓ Tiếp tuyến: Kiểm tra format (tangent M (circle O) AB)
   → M là tiếp điểm, O là tâm, AB là chuỗi 2 ký tự tạo đường thẳng
   → 🔥 PHẢI có (segment O M) NGAY SAU (tangent ...) - bán kính từ tâm đến tiếp điểm!
   → Ngoại lệ duy nhất: M đã nằm trên đường kính đã được vẽ sẵn

13. ✓ Hình chiếu: Kiểm tra segment được vẽ từ ĐIỂM GỐC đến HÌNH CHIẾU
   → (projection O (segment A C)) → PHẢI có (segment O H) - TỪ O ĐẾN H!
   → KHÔNG ĐƯỢC (segment A H) - SAI!

14. ✓ Khi có yêu cầu chứng minh về độ dài: PHẢI vẽ các segment tương ứng
   → "Chứng minh AC = AD" → PHẢI có (segment A C) và (segment A D)
   → "Chứng minh AB = CD" → PHẢI có (segment A B) và (segment C D)

15. ✓ 🔥 QUY TẮC VÀNG - VẼ SEGMENT KHI ĐỀ NHẮC ĐẾN:
   → TÌM TẤT CẢ CẠNH/ĐOẠN THẲNG trong đề bài (AC, AM, BC, AO, OH, MN, ...)
   → VỚI MỖI cạnh được nhắc đến: KIỂM TRA xem đã có (segment ...) chưa?
   → Nếu CHƯA CÓ → THÊM NGAY (segment X Y)
   • Ví dụ: "đề nhắc AO" → cần (segment A O)
   • Ví dụ: "đề nhắc AM vuông góc BC" → cần (segment A M) và (segment B C)
   • Ví dụ: "AO là đường trung trực BC" → cần (segment A O) và (segment B C)
   • Ví dụ: "kẻ OH", "nối AC" → cần (segment O H), (segment A C)

16. ✓ ĐƯỜNG KÍNH: (diameter M N O) + PHẢI có (segment M N)
   → (diameter M N O) chỉ là constraint, KHÔNG tự động vẽ
   → Nếu đề đề cập đến đường kính → PHẢI thêm (segment M N)
   → Ví dụ: "tiếp tuyến vuông góc MN" → cần cả (diameter M N O) và (segment M N)

17. ✓ Không có comment # trong DSL output
   → Chỉ có DSL thuần túy với \n

18. ✓ Field "instruction" là bài toán gốc tiếng Việt, KHÔNG được thay đổi
   → Copy nguyên văn đề bài, không dịch sang tiếng Anh

19. ✓ Kiểm tra lại lần cuối: Có điểm nào được define 2 lần không?

20. ✓ 🔥 SAU MỌI (tangent ...) → KIỂM TRA CÓ (segment O <tiếp-điểm>) NGAY SAU KHÔNG?
   → Nếu KHÔNG CÓ và tiếp điểm KHÔNG nằm trên đường kính → THÊM NGAY!
   → Đây là lỗi PHỔ BIẾN NHẤT - hình sẽ KHÔNG VẼ ĐƯỢC nếu thiếu!
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

⚠️ TRƯỚC KHI OUTPUT, KIỂM TRA HÌNH CHIẾU VÀ VUÔNG GÓC:

Bước 1: Tìm từ khóa hình chiếu/vuông góc trong đề
• "Kẻ OH vuông góc AC (H ∈ AC)"
• "H là hình chiếu O lên AC"
• "OH ⊥ AC"

Bước 2: Kiểm tra cách xử lý
• PHẢI dùng: (define H point (projection O (segment A C)))
• KHÔNG dùng: (define H point) + (perpendicular ...) + (on-segment ...)

Bước 3: Kiểm tra segment được vẽ ĐÚNG
• PHẢI có: (segment O H) - từ ĐIỂM GỐC (O) đến HÌNH CHIẾU (H)
• TUYỆT ĐỐI KHÔNG: (segment A H) - SAI HOÀN TOÀN!

⚠️ TRƯỚC KHI OUTPUT, KIỂM tra SEGMENT CẦN THIẾT:

Bước 1: ĐỌC KỸ ĐỀ BÀI - Liệt kê TẤT CẢ cạnh/đoạn thẳng được nhắc đến
• "AC = AD" → cạnh AC, AD
• "AB ⊥ CD" → cạnh AB, CD
• "AO là đường trung trực BC" → cạnh AO, BC
• "AM vuông góc BC" → cạnh AM, BC
• "kẻ OH", "nối AC" → cạnh OH, AC
• "đường kính MN" → cạnh MN

Bước 2: KIỂM TRA TỪNG CẠNH - Đã có (segment ...) chưa?
VÍ DỤ: Đề nhắc "AC", "AD", "AO"
• Tìm (segment A C) trong DSL → Nếu CHƯA có → THÊM ngay!
• Tìm (segment A D) trong DSL → Nếu CHƯA có → THÊM ngay!
• Tìm (segment A O) trong DSL → Nếu CHƯA có → THÊM ngay!

Bước 3: ĐẶC BIỆT QUAN TRỌNG
• Tiếp tuyến AB → segment A B đã có rồi
• Dây CD → segment C D đã có rồi
• Đường kính MN → PHẢI THÊM (segment M N) - vì (diameter M N O) KHÔNG tự vẽ!
• Triangle ABC → segment A B, B C, C A đã có rồi

⚠️ TRƯỚC KHI OUTPUT, KIỂM TRA ĐƯỜNG KÍNH VÀ SEGMENT:

Bước 1: Tìm tất cả quan hệ hình học được mô tả
• "Chứng minh AC = AD" → Vẽ AC = AD → Cần (equal-distance A C A D)
• "Chứng minh AB ⊥ CD" → Vẽ AB ⊥ CD → Cần (perpendicular (segment A B) (segment C D))
• "Chứng minh H là trung điểm AC" → Vẽ H ở giữa AC → Cần (define H point (midpoint A C))

Bước 2: Kiểm tra các segment cần vẽ
• "AC = AD" → CẦN (segment A C) và (segment A D)
• "AB ⊥ CD" → CẦN (segment A B) và (segment C D)
• "AB ⊥ CD" → CẦN (segment A B) và (segment C D) - KHÔNG thêm perpendicular!
• "H là trung điểm" → CẦN segment trước midpoint

⚠️ TRƯỚC KHI OUTPUT, KIỂM TRA ĐƯỜNG KÍNH VÀ SEGMENT:

Bước 1: Tìm từ khóa đường kính trong đề
• "có đường kính MN"
• "đường kính AB"
• "tiếp tuyến vuông góc với MN"

Bước 2: Kiểm tra DSL
• PHẢI có: (diameter M N O) - constraint
• PHẢI có: (segment M N) - vẽ đường kính như đoạn thẳng
• KHÔNG ĐƯỢC thiếu (segment M N)!

BưỚC 3: Kiểm tra các đoạn thẳng được đề cập
• ĐỌC KỸ ĐỀ BÀI - tìm TẤT CẢ các cạnh/đoạn thẳng được nhắc đến:
  - "AC = AD" → nhắc AC, AD
  - "AB ⊥ CD" → nhắc AB, CD  
  - "AO là đường trung trực BC" → nhắc AO, BC
  - "AM vuông góc BC" → nhắc AM, BC
  - "kẻ OH", "nối AC" → nhắc OH, AC
  - "đường kính MN" → nhắc MN

• VỚI MỔI cạnh được nhắc (ví dụ AC):
  - Kiểm tra DSL đã có (segment A C) chưa?
  - Nếu CHƯẢ → THÊM NGAY (segment A C)
  - Nếu ĐÃ CÓ → Bỏ QUA

• ĐẶC BIỆT:
  - Tiếp tuyến AB → đã có (segment A B) rồi
  - Dây CD → đã có (segment C D) rồi
  - Đường kính MN → PHẢI THÊM (segment M N)

Bước 3: Kiểm tra bán kính cho tiếp tuyến
• Tiếp tuyến tại A → CẦN (segment O A) - bán kính từ tâm đến tiếp điểm
• Trừ khi A nằm trên đường kính → bán kính đã có sẵn

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

5. Tìm quan hệ hình học trong đề bài → Kiểm tra có thêm constraint chưa?
   • "Chứng minh AB = AC" → PHẢI có (equal-distance A B A C) hoặc (segment A B) và (segment A C) với giá trị bằng nhau
   • "Chứng minh góc AOC = góc BAC" → PHẢI có (angle-equal A O C B A C)
   • "Chứng minh AB ⊥ CD" → PHẢI có (perpendicular (segment A B) (segment C D))
   • Nếu thiếu → THÊM constraint tương ứng

═══ INPUT ═══
{{ extract }}

"""
