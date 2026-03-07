prompt: str = """
Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL.

LƯU Ý VỀ TÊN ĐIỂM TRONG VÍ DỤ - CỰC KỲ QUAN TRỌNG:
TẤT CẢ VÍ DỤ TRONG FILE NÀY DÙNG TÊN ĐIỂM CỤ THỂ (A, B, C, H, M, I, O, N, P...) CHỈ LÀ VÍ DỤ MINH HỌA!
BẠN PHẢI THAY THẾ BẰNG TÊN ĐIỂM TRONG ĐỀ BÀI!

- VÍ dụ dùng "H" nhưng đề gọi là "K" → Dùng "K"
- Ví dụ dùng "M" nhưng đề gọi là "I" → Dùng "I" 
- Ví dụ dùng "C, D" cho dây nhưng đề gọi là "E, F" → Dùng "E, F"
- NGUYÊN TẮC: ĐỌC ĐỀ BÀI, DÙNG ĐÚNG TÊN ĐIỂM TRONG ĐỀ!

5 LỖI CẤM TUYỆT ĐỐI - ĐỌC TRƯỚC KHI LÀM 

1. (on-segment A B) - CHỈ 2 ĐIỂM → SAI!
   (on-segment M A B) - CẦN ĐÚNG 3 ĐIỂM
   • Cú pháp: (on-segment <điểm-nằm-giữa> <endpoint1> <endpoint2>)
   • VÍ DỤ: M nằm giữa A và B → (on-segment M A B)
   • ĐẶC BIỆT VỚI TIẾP TUYẾN:
     - "AB là tiếp tuyến tại A" → A là endpoint → KHÔNG DÙNG (on-segment A B)
     - "Từ A kẻ tiếp tuyến AM" → M là endpoint → KHÔNG DÙNG (on-segment A M)
     - "AB tiếp xúc tại M (M giữa A và B)" → CẦN (on-segment M A B)

2. (define M point) ... (define M point) - DEFINE TRÙNG → SAI!
   MỖI ĐIỂM CHỈ DEFINE MỘT LẦN DUY NHẤT
   • Nếu M đã được define → KHÔNG BAO GIỜ define M lại
   • Tìm trong toàn bộ DSL trước khi define điểm mới

3. Tiếp điểm nhưng THIẾU (on-circle M O) → SAI!
   (define M point) → (on-circle M O) → (tangent M (circle O) AB)
   • Tiếp điểm LUÔN LUÔN nằm trên đường tròn
   • Không có on-circle → M có thể rơi ra ngoài đường tròn

4. "Lấy dây CD" nhưng THIẾU (on-circle C O) và (on-circle D O) → SAI!
   Dây cung → CẢ 2 ĐẦU đều phải on-circle
   (define C point)
   (on-circle C O)  ← BẮT BUỘC!
   (define D point)
   (on-circle D O)  ← BẮT BUỘC!
   (segment C D)

5. VẼ TẤT CẢ MÔ TẢ HÌNH HỌC TRONG ĐỀ BÀI!
   • Đề: "Chứng minh AB = AC" → VÉ cả AB và AC bằng nhau → THÊM (equal-distance A B A C)
   • Đề: "Chứng minh góc A = 60°" → VẼ góc A = 60° → THÊM (angle-measure B A C 60)
   • Đề: "Chứng minh AB ⊥ CD" → VẼ AB vuông góc CD → THÊM (perpendicular (segment A B) (segment C D))
   • Đề: "Chứng minh ∠BAC = ∠OCA" → VẼ cả 2 góc bằng nhau → THÊM (angle-equal B A C O C A)
   • NGUYÊN TẮC: Miễn đề BÀI MÔ TẢ quan hệ hình học → VẼ LUÔN, không phân biệt "cho" hay "chứng minh"

QUY TẮC VÀNG - VẼ SEGMENT KHI ĐỀ NHẮC ĐẾN:

ĐỌC KỸ ĐỀ BÀI - TÌM TẤT CẢ CẠNH/ĐOẠN THẲNG ĐƯỢC NHẮC ĐẾN
• "AC = AD" → nhắc AC, AD
• "AB ⊥ CD" → nhắc AB, CD
• "AO là đường trung trực BC" → nhắc AO, BC
• "AM vuông góc BC" → nhắc AM, BC
• "kẻ OH", "nối AC" → nhắc OH, AC
• "đường kính MN" → nhắc MN

VỚI MỖI CẠNH ĐƯỢC NHẮC (ví dụ AC):
1. Kiểm tra DSL đã có (segment A C) chưa?
2. Nếu CHƯA → THÊM NGAY (segment A C)
3. Nếu ĐÃ CÓ → Bỏ qua

ĐẶC BIỆT:
• Tiếp tuyến AB → đã có (segment A B) rồi
• Dây CD → đã có (segment C D) rồi
• Đường kính MN → PHẢI THÊM (segment M N) - vì (diameter M N O) không tự vẽ!

═══ CÚ PHÁP DSL ═══

1. HÌNH
   • (triangle (A B C)) / (triangle (A B C) (isosceles A)) / (triangle (A B C) (right B)) / (triangle (A B C) (equilateral))
   • (square (A B C D))
   • (rectangle (A B C D))
   • (trapezoid (A B C D))
   • (parallelogram (A B C D))
   • (rhombus (A B C D))

2. ĐIỂM: (define <name> point <construction>)
   • Đặc biệt: (midpoint B C), (centroid A B C), (incenter A B C), (circumcenter A B C), (orthocenter A B C)
   • Hình chiếu: (projection A (segment B C))
   • Phân giác: (bisector B A C) - góc BAC, đỉnh A
   • Giao điểm: (inter-ll C D A B) - giao của CD và AB
   • Tự do: (segment A B), (line A B)
   
   HÌNH CHIẾU + VUÔNG GÓC - CỰC QUAN TRỌNG:
   
   ĐỊNH NGHĨA TỔNG QUÁT:
   Hình chiếu của một điểm lên một đường thẳng là điểm nằm trên đường thẳng sao cho đoạn thẳng nối từ điểm gốc đến hình chiếu vuông góc với đường thẳng đó.
   
   KHI NÀO LÀ HÌNH CHIẾU? - CỰC KỲ QUAN TRỌNG:
   • "Kẻ OH vuông góc AC (H ∈ AC)" → ĐÂY LÀ HÌNH CHIẾU!
   • "H là hình chiếu của O lên AC" → ĐÂY LÀ HÌNH CHIẾU!
   • "Kẻ OH ⊥ AC, H nằm trên AC" → ĐÂY LÀ HÌNH CHIẾU!
   • "Từ O hạ đường vuông góc xuống AC, giao điểm là H" → ĐÂY LÀ HÌNH CHIẾU!
      ⚠️ AC Ở ĐÂY LÀ GÌ? (trong ngữ cảnh đường tròn tâm O):
      • AC có thể là DÂY CUNG (A và C nằm trên đường tròn)
      • AC có thể là ĐƯỜNG KÍNH (A và C là 2 đầu đường kính)
      • AC có thể là ĐOẠN THẲNG BẤT KỲ (A, C là 2 điểm trong hình)
      • AC có thể là TIẾP TUYẾN (nếu đề nói "AC là tiếp tuyến")
      → QUAN TRỌNG: Dù AC là gì, khi "kẻ OH ⊥ AC" = H là HÌNH CHIẾU của O lên đường thẳng AC!
   
   CÚ PHÁP: (define H point (projection O (segment A C)))
   • O: điểm gốc cần chiếu
   • (segment A C): đường thẳng cần chiếu lên
   • H: hình chiếu (tự động nằm trên AC và OH ⊥ AC)
   
   QUY TẮC: DÙNG PROJECTION, KHÔNG define H riêng + perpendicular
   • SAI: (define H point) + (perpendicular ...) + (on-segment H A C)
   • ĐÚNG: (define H point (projection O (segment A C)))
   
   🔥🔥 PROJECTION ĐÃ BAO GỒM PERPENDICULAR - KHÔNG THÊM NỮA!
   • Projection TỰ ĐỘNG làm H nằm trên AC
   • Projection TỰ ĐỘNG làm OH ⊥ AC
   • TUYỆT ĐỐI KHÔNG THÊM: (perpendicular (segment O H) (segment A C))
   • THÊM perpendicular = THỪA = CÓ THỂ GÂY XUNG ĐỘT!
   
   QUY TẮC VẼ SEGMENT: LUÔN vẽ từ ĐIỂM GỐC đến HÌNH CHIẾU
   • PHẢI có: (segment O H) - từ điểm gốc O đến hình chiếu H
   • SAI: (segment A H) - không vẽ từ điểm khác!
   
   CỰC KỲ QUAN TRỌNG - TUYỆT ĐỐI KHÔNG VẼ THỪA:
   Khi H là hình chiếu lên AC:
   • H ĐÃ TỰ ĐỘNG NẰM TRÊN AC rồi (do projection)
   • AC đã được vẽ bằng (segment A C)
   • TUYỆT ĐỐI KHÔNG vẽ thêm (segment A H) hoặc (segment H C)
   • VẼ THÊM = VẼ ĐÈ = SAI HOÀN TOÀN!
   
   CHỈ VẼ:
   1. (segment A C) - đoạn gốc
   2. (define H point (projection O (segment A C))) - hình chiếu
   3. (segment O H) - từ điểm gốc đến hình chiếu
   
   KHÔNG BAO GIỜ VẼ:
   • (segment A H) 
   • (segment H C) 
   • (segment M H) với M bất kỳ trên AC
   • (perpendicular (segment O H) (segment A C)) ← THỪA! Projection đã có sẵn!
   • (define M point (midpoint A C)) nếu đề KHÔNG nhắc M ← Không tự ý define điểm! 
   
   VÍ DỤ CỤ THỂ 1: "Kẻ OH vuông góc AC (H ∈ AC)"
   (segment A C)        ← Phải có segment trước!
   (define H point (projection O (segment A C)))
   (segment O H)        ← Vẽ từ ĐIỂM GỐC (O) đến HÌNH CHIẾU (H)
   ← CHỈ 3 DÒNG! KHÔNG thêm perpendicular (projection đã vuông góc rồi)!
   
   VÍ DỤ CỤ THỂ 2: "Kẻ OH ⊥ AC (H ∈ AC). Chứng minh H là trung điểm AC"
   (segment A C)
   (define H point (projection O (segment A C)))
   (segment O H)
   ← CHỈ 3 DÒNG! KHÔNG define M! KHÔNG thêm perpendicular!
   ← Đề nói "H là trung điểm" chứ không nói "M là trung điểm" → chỉ có H!
   
   VÍ DỤ CỤ THỂ 2: "H là hình chiếu B lên AC"
   (define H point (projection B (segment A C)))
   (segment B H)        ← Từ B đến H!
   
   VÍ DỤ CỤ THỂ 3: "H là hình chiếu O lên AC"
   (define H point (projection O (segment A C)))
   (segment O H)        ← Từ O đến H!
   
   GIAO ĐIỂM: "CD cắt AB tại H" → (define H point (inter-ll C D A B))
   
   ⚠️ LƯU Ý: H, O, A, C, B trong các ví dụ trên là TÊN VÍ DỤ - thay bằng tên điểm thực tế trong đề bài!
   
   CẤM DÙNG on-circle TRONG CONSTRUCTION:
   SAI: (define C point (on-circle C O))
   ĐÚNG: (define C point) + (on-circle C O)
   
   TRUNG ĐIỂM CUNG vs TRUNG ĐIỂM ĐOẠN:
   
   ⚠️ LƯU Ý: M, A, B, C trong ví dụ là TÊN VÍ DỤ - Đề dùng tên khác thì thay tên tương ứng!
   
   ĐỊNH NGHĨA:
   • TRUNG ĐIỂM CUNG: Điểm nằm trên đường tròn chia cung thành hai cung bằng nhau
   • TRUNG ĐIỂM ĐOẠN: Điểm nằm giữa hai đầu mút của đoạn thẳng và cách đều hai đầu
   
   PHÂN BIỆT:
   • Trung điểm CUNG → Điểm TỰ DO trên đường tròn → KHÔNG dùng midpoint
   • Trung điểm ĐOẠN → Điểm giữa đoạn thẳng → DÙNG midpoint
   
   VÍ DỤ TRUNG ĐIỂM CUNG: "C là trung điểm cung AB"
   (define C point)
   (on-circle C O)      ← C tự do trên đường tròn
   
   VÍ DỤ TRUNG ĐIỂM ĐOẠN: "M là trung điểm đoạn AB"
   (define M point (midpoint A B))
   
   ⚠️ LƯU Ý: M, A, B, C trong ví dụ là TÊN VÍ DỤ - Đề dùng tên khác thì thay tên tương ứng!
   
   ĐƯỜNG TRUNG TRỰC - CỰC KỲ QUAN TRỌNG:
   
   ĐỊNH NGHĨA TỔNG QUÁT:
   Đường trung trực của một đoạn thẳng là đường thẳng đi qua trung điểm của đoạn thẳng đó và vuông góc với đoạn thẳng đó.
   
   PHÂN TÍCH VÍ DỤ: Khi đề nói "AO là đường trung trực của BC"
   • AO là đường thẳng (đường trung trực) - đã tồn tại trước
   • BC là đoạn thẳng bị chia
   • AO đi qua trung điểm M của BC và AO ⊥ BC
   • M nằm trên AO (M xuất hiện sau, do đường trung trực đi qua trung điểm)
   
   QUY TẮC VỀ ĐIỂM GIAO/TRUNG ĐIỂM - CỰC KỲ QUAN TRỌNG:
   
   QUY TẮC NÀY ÁP DỤNG CHO MỌI TÊN ĐIỂM: M, I, H, N, P, K, v.v.
   (Dưới đây dùng M làm ví dụ, nhưng áp dụng cho BẤT KỲ TÊN ĐIỂM NÀO trong đề)
   
   NGUYÊN TẮC: CHỈ DEFINE ĐIỂM KHI ĐỀ BÀI NÓI RÕ VỀ ĐIỂM ĐÓ! 🔥🔥🔥
   
   DEFINE ĐIỂM KHI ĐỀ CÓ CÁC CỤM TỪ (X = tên điểm bất kỳ):
   • "Gọi X là trung điểm BC" / "Gọi M là trung điểm BC"
   • "X là giao điểm của AO và BC" / "I là giao điểm..."
   • "Gọi X là giao điểm" / "Gọi H là giao điểm"
   • "Lấy X là trung điểm" / "Lấy N là trung điểm"
   • "Điểm X nằm trên BC sao cho..." / "Điểm P nằm trên BC..."
   
   TUYỆT ĐỐI KHÔNG DEFINE ĐIỂM KHI:
   • Đề CHỈ nói "AO là đường trung trực BC" - KHÔNG NHẮC TÊN ĐIỂM NÀO
   • Đề CHỈ nói "Chứng minh AO là đường trung trực" - KHÔNG NHẮC TÊN ĐIỂM
   • Đề KHÔNG có tên điểm đó trong toàn bộ đề bài - KHÔNG VẼ!
   
   CÁCH DEFINE ĐIỂM (khi đề yêu cầu) - ví dụ với M:
   • KHI có đường trung trực: DÙNG (inter-ll A O B C) - M là GIAO ĐIỂM
   • KHI không có đường trung trực: DÙNG (midpoint B C) - M là trung điểm đơn thuần
   • LÝ do inter-ll: Chấm nằm CHÍNH XÁC tại giao điểm, không lệch
   • LÝ do midpoint: Chỉ dùng khi điểm là điểm riêng, không phải giao điểm
   
   THỨ TỰ VẼ ĐÚNG (TRÁNH VẼ ĐÈ):
   
   1. VẼ ĐOẠN BC trước:
      (segment B C)
   
   2. VẼ ĐƯỜNG TRUNG TRỰC AO:
      (segment A O)
      
   3. THÊM CONSTRAINT vuông góc:
      (perpendicular (segment A O) (segment B C))
   
   4. CHỈ KHI ĐỀ NÓI RÕ VỀ ĐIỂM GIAO - DEFINE ĐIỂM:
      
      🔍 KIỂM TRA ĐỀ BÀI CÓ NHẮC TÊN ĐIỂM GIAO KHÔNG? (M, I, H, N, P, K...)
      • Tìm trong đề: "Gọi [tên]", "[tên] là", "điểm [tên]", "trung điểm [tên]"
      • VÍ DỤ: "Gọi M", "M là", "I là giao điểm", "điểm H", "N nằm trên"
      • Nếu KHÔNG TÌM THẤY TÊN ĐIỂM NÀO → DỪNG LẠI, KHÔNG DEFINE!
      • Nếu TÌM THẤY → Tiếp tục define điểm đó
      
      CÁCH DEFINE M (khi đề nhắc):
      • PHẢI DÙNG: (define M point (inter-ll A O B C))  ← inter-ll!
      • KHÔNG DÙNG: (define M point (midpoint B C))  ← SAI! Chấm lệch!
      • Lý do: inter-ll cho chấm M nằm CHÍNH XÁC tại giao AO và BC
      
   CỰC KỲ QUAN TRỌNG - TUYỆT ĐỐI KHÔNG VẼ THỪA:
   Khi chứng minh AO là đường trung trực của BC:
   • M tự động nằm trên AO (do AO đi qua trung điểm BC)
   • ĐÃ VẼ (segment A O) rồi → M nằm sẵn trên đó
   • TUYỆT ĐỐI KHÔNG vẽ thêm (segment A M) hoặc (segment M O)
   • VẼ THÊM = VẼ ĐÈ = SAI HOÀN TOÀN!
   
   CHỈ VẼ 3 THÀNH PHẦN:
   1. (segment B C) - đoạn bị chia
   2. (segment A O) - đường trung trực
   3. (perpendicular (segment A O) (segment B C))
   
   KHÔNG BAO GIỜ VẼ:
   • (segment A M) 
   • (segment M O) 
   • (segment M X) với X bất kỳ 
   
   VÍ DỤ ĐÚNG 1: "Chứng minh AO là đường trung trực của BC" (KHÔNG nhắc tên điểm nào)
   (segment B C)
   (segment A O)                                  ← Vẽ đường trung trực
   (perpendicular (segment A O) (segment B C))    ← AO ⊥ BC
   ← CHỈ 3 DÒNG! KHÔNG THÊM GÌ NỮA!
   
   VÍ DỤ ĐÚNG 2: "M là trung điểm BC. Chứng minh AO là đường trung trực BC" (ĐỀ NHẮC M)
   (segment B C)
   (segment A O)
   (perpendicular (segment A O) (segment B C))
   (define M point (inter-ll A O B C))           ← DÙNG inter-ll cho chấm chính xác!
   
   VÍ DỤ ĐÚNG 3: "Gọi I là giao điểm của AO và BC" (ĐỀ NHẮC I thay vì M)
   (segment B C)
   (segment A O)
   (perpendicular (segment A O) (segment B C))
   (define I point (inter-ll A O B C))           ← Đổi tên thành I - quy tắc giống hệt M!
   
   VÍ DỤ SAI - TUYỆT ĐỐI TRÁNH:
   (segment B C)
   (define M point (midpoint B C))  ← SAI! Không cần define M
   (segment A M)  ← SAI! VẼ ĐÈ
   (segment A O)
   (perpendicular (segment A O) (segment B C))
   
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
   
   ⚠️ LƯU Ý: TẤT CẢ TÊN ĐIỂM TRONG PHẦN NÀY (M, N, C, D, A, B, O...) CHỈ LÀ VÍ DỤ!
   Đề bài dùng tên khác → THAY NGAY bằng tên đó! Ví dụ: Đề gọi "E, F" thì dùng E, F thay vì C, D
   
   ĐỊNH NGHĨA:
   • ĐƯỜNG KÍNH: Dây đi qua tâm đường tròn, chia đường tròn thành hai nửa bằng nhau
   • DÂY: Đoạn thẳng nối hai điểm bất kỳ trên đường tròn (không nhất thiết qua tâm)
   
   **ĐƯỜNG KÍNH** (đi qua tâm O):
   CÚ PHÁP: (diameter A B O)
   • A, B: 2 điểm đầu mút của đường kính
   • O: tâm đường tròn (tự động nằm giữa A và B)
   • Tự động tạo: on-circle A O, on-circle B O, on-segment O A B
   
   LƯU Ý QUAN TRỌNG:
   • (diameter M N O) CHỈ là CONSTRAINT - KHÔNG tự động vẽ segment!
   • Nếu đề đề cập đến đường kính như đường thẳng → PHẢI thêm (segment M N)
   
   **DÂY THƯỜNG** (không qua tâm):
   3 BƯỚC BẮT BUỘC:
   1. Định nghĩa điểm: (define C point) + (define D point)
   2. Đặt trên đường tròn: (on-circle C O) + (on-circle D O)
   3. Vẽ đoạn thẳng: (segment C D)
   
   LỖI PHỔ BIẾN VỀ DÂY:
   • SAI: (define C point) + (define D point) + (segment C D) → THIẾU on-circle!
   • SAI: (segment C D) + (on-circle C O) + (on-circle D O) → THIẾU define!
   • ĐÚNG: Làm đủ 3 bước như trên
   
   VÍ DỤ CỤ THỂ: "Lấy dây CD song song với AB"
   (define C point)         ← Bước 1: Define điểm C
   (on-circle C O)          ← Bước 2: C trên đường tròn
   (define D point)         ← Bước 1: Define điểm D
   (on-circle D O)          ← Bước 2: D trên đường tròn
   (segment C D)            ← Bước 3: Vẽ dây CD
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
      ⚠️ KHÔNG VẼ (segment O M) - M đã trên đường kính MN, bán kính OM đã có!
      ⚠️ KHÔNG VẼ (segment O M) - M đã trên đường kính MN, bán kính OM đã có!

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
   
   ⚠️ LƯU Ý: TẤT CẢ TÊN ĐIỂM TRONG PHẦN NÀY (A, B, C, M, O...) CHỈ LÀ VÍ DỤ!
   Đề dùng tên khác → THAY bằng tên đó! Ví dụ: Tiếp điểm "P" thì dùng P, không phải M
   
   QUY TẮC VÀNG: 
   • Tiếp điểm là ENDPOINT → KHÔNG dùng on-segment (phổ biến 90% trường hợp)
   • Mỗi điểm CHỈ define 1 LẦN
   • on-segment CẦN ĐÚNG 3 điểm khác nhau
   
   ĐỊNH NGHĨA TIẾP TUYẾN:
   Tiếp tuyến của đường tròn là đường thẳng chỉ tiếp xúc (chạm) với đường tròn tại duy nhất một điểm, gọi là tiếp điểm.
   
   TÍNH CHẤT:
   • Tiếp tuyến KHÔNG CẮT đường tròn
   • Tiếp tuyến VUÔNG GÓC với bán kính tại tiếp điểm
   
   ĐỊNH NGHĨA TIẾP ĐIỂM:
   Tiếp điểm là điểm duy nhất mà tiếp tuyến và đường tròn tiếp xúc nhau.
   
   TÍNH CHẤT TIẾP ĐIỂM (phải thỏa mãn đồng thời):
   1. Nằm trên đường tròn
   2. Nằm trên đường thẳng tiếp tuyến  
   3. Bán kính từ tâm đến tiếp điểm vuông góc với tiếp tuyến
   
   KÝ HIỆU: M là tiếp điểm của AB với (O) → M ∈ (O), M ∈ AB, OM ⊥ AB
   
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
   4. NẾU đề nói "A nằm ngoài đường tròn": PHẢI thêm (distance O A <giá_trị>)
      • Với đường tròn bán kính R, chọn khoảng cách > R (ví dụ: R=0.5 → chọn 1.0 hoặc 1.5)
      • Nếu không có bán kính cụ thể, dùng 1.5 (giả sử bán kính mặc định ~1.0)
   
   4b. 🔥🔥🔥 CONSTRAINT CHO ĐIỂM THỨ 2 TRÊN TIẾP TUYẾN - CỰC KỲ QUAN TRỌNG!
      • Khi có tiếp tuyến AB (A là tiếp điểm, B là điểm thứ 2):
      • NẾU đề KHÔNG nói gì về B (không nói khoảng cách, không nói vị trí cụ thể)
      • B KHÔNG CÓ CONSTRAINT → B sẽ TỰ DO → Tiếp tuyến KHÔNG ỔN ĐỊNH!
      • 🚨 GIẢI PHÁP: PHẢI THÊM (equal-distance A B 1.0) hoặc (distance A B 1.0)
      • Điều này cho AB có độ dài cố định → Tiếp tuyến ổn định!
      
      VÍ DỤ:
      ❌ SAI - B tự do:
      (define B point)           ← Không constraint!
      (segment A B)
      (tangent A (circle O) AB)  ← B có thể ở BẤT KỲ ĐÂU → không ổn định!
      
      ✅ ĐÚNG - B có constraint:
      (define B point)
      (equal-distance A B 1.0)   ← ✅ Cho AB có độ dài cố định!
      (segment A B)
      (tangent A (circle O) AB)  ← ✅ Tiếp tuyến ổn định!
      
      ⚠️ NGOẠI LỆ: Nếu đề NÓI RÕ về B ("B nằm trên...", "AB = ...") → theo đề
   
   5. Vẽ segment tiếp tuyến: (segment A M) hoặc (segment A B)
   6. CHỈ KHI M NẰM GIỮA A và B (trường hợp hiếm): (on-segment M A B)
      → Nếu M là ENDPOINT của segment → BỎ QUA bước này!
   7. Khai báo tiếp tuyến: (tangent M (circle O) AM) hoặc (tangent M (circle O) AB)
   
   8. 🔥🔥🔥 VẼ BÁN KÍNH (segment O M) - DÒNG TIẾP THEO NGAY SAU (tangent ...)! 🔥🔥🔥
      
      ⚠️ QUY TẮC TUYỆT ĐỐI: (segment O M) PHẢI Ở DÒNG NGAY SAU (tangent M ...)
      ⚠️ KHÔNG ĐƯỢC CÓ BẤT KỲ DÒNG NÀO CHEN GIỮA!
      ⚠️ KHÔNG ĐƯỢC define điểm khác, không được vẽ gì khác trước khi vẽ (segment O M)!
      
      VÍ DỤ:
      (tangent M (circle O) MA)
      (segment O M)              ← 🔥 NGAY DÒNG NÀY!
      (define C point)           ← ✅ Điểm khác SAU ĐÓ
      
      KHÔNG ĐƯỢC:
      (tangent M (circle O) MA)
      (define C point)           ← ❌ SAI! Không được chen vào!
      (segment O M)              ← ❌ Quá muộn!
      
      ⚠️⚠️ NGOẠI LỆ DUY NHẤT - KHÔNG VẼ BÁN KÍNH KHI:
      • M đã nằm trên ĐƯỜNG KÍNH được định nghĩa trước đó
      • Ví dụ: (diameter M N O) hoặc (diameter A M O) đã có → BỎ QUA (segment O M)
      • Lý do: Bán kính OM đã tồn tại trong đường kính, vẽ lại sẽ trùng lặp!
      
      ✅ VẼ BÁN KÍNH KHI:
      • M được define riêng rẽ, KHÔNG phải điểm đầu mút đường kính
      • M là tiếp điểm tự do trên đường tròn
   
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
   (define M point (midpoint B C))  ← Dùng midpoint vì AM là đường riêng!
   (segment A M)              ← VẼ AM như 1 đoạn riêng
   (perpendicular (segment A M) (segment B C))  ← AM ⊥ BC
   
   LƯU Ý: Ví dụ này dùng midpoint vì AM là đường RIÊNG, không phải đường trung trực!
   
   VÍ DỤ 4_alt: "Từ A ngoài (O) kẻ AB, AC (B, C tiếp điểm). Chứng minh AO là đường trung trực BC"
   ... (giống như trên đến segment B C)
   (segment B C)
   (segment A O)              ← VẼ đường trung trực AO
   (perpendicular (segment A O) (segment B C))
   ← CHỈ 3 dòng cuối! KHÔNG define M vì đề không nhắc!
   
   VÍ DỤ 4_alt2: "Từ A ngoài (O) kẻ AB, AC. M là giao AO và BC"
   ... (giống như trên đến segment B C)
   (segment B C)
   (segment A O)
   (perpendicular (segment A O) (segment B C))
   (define M point (inter-ll A O B C))  ← Dùng inter-ll vì M là GIAO ĐIỂM!

VÍ DỤ 4b: "AB là tiếp tuyến tại A, AC là dây. Chứng minh ∠BAC = ∠OCA"
   (define O point)
   (circle O)
   (define A point)
   (on-circle A O)            ← A là tiếp điểm
   (define B point)
   (equal-distance A B 1.0)   ← 🔥 BẮT BUỘC: Constraint cho B để tiếp tuyến ổn định!
   (segment A B)
   (tangent A (circle O) AB)
   (segment O A)              ← 🔥🔥 BẮT BUỘC: Vẽ bán kính OA Ở DÒNG NÀY - NGAY SAU tangent!
   (define C point)
   (on-circle C O)            ← C trên đường tròn
   (segment A C)              ← Vẽ dây AC
   (segment O C)              ← BẮT BUỘC: Vẽ bán kính OC để có tam giác OAC
   (angle-equal B A C O C A)  ← VẼ ∠BAC = ∠OCA theo đề bài

   🚨🚨🚨 LỖI CỰC KỲ THƯỜNG GẶP VÀ PHẢI TRÁNH:
   
   ❌ LỖI 1 - THIẾU CONSTRAINT CHO B:
   (define B point)           ← ❌ B không có constraint!
   (segment A B)
   (tangent A (circle O) AB)
   
   → HẬU QUẢ: B tự do → Tiếp tuyến không ổn định!
   → GIẢI PHÁP: Thêm (equal-distance A B 1.0) sau (define B point)
   
   ❌ LỖI 2 - SAI THỨ TỰ (segment O A):
   (tangent A (circle O) AB)
   (define C point)           ← ❌ SAI! Không được chen bất kỳ dòng nào!
   (on-circle C O)
   (segment A C)
   (segment O A)              ← ❌ SAI VỊ TRÍ! Quá muộn rồi!
   
   → HẬU QUẢ: Optimizer sẽ đặt sai vị trí A, làm AB không còn là tiếp tuyến!
   → TRIỆU CHỨNG: AB cắt đường tròn thay vì chỉ chạm tại A
   → ĐÂY LÀ LỖI NGHIÊM TRỌNG NHẤT - HÌNH VẼ SẼ HOÀN TOÀN SAI!
   
   ✅ ĐÚNG - LUÔN LUÔN LÀM THẾ NÀY:
   (define B point)
   (equal-distance A B 1.0)   ← ✅ Constraint cho B!
   (segment A B)
   (tangent A (circle O) AB)
   (segment O A)              ← ✅ Ở NGAY DÒNG TIẾP THEO - TUYỆT ĐỐI KHÔNG CÓ GÌ CHEN GIỮA!
   (define C point)           ← ✅ Các điểm khác MỚI được define sau đó
   (on-circle C O)
   (segment A C)
   
   🔥🔥🔥 QUY TẮC VÀNG - GHI NHỚ SUỐT ĐỜI:
   (segment O X) PHẢI XUẤT HIỆN Ở DÒNG NGAY SAU (tangent X ...)
   KHÔNG CÓ BẤT KỲ DÒNG NÀO CHEN GIỮA - TUYỆT ĐỐI KHÔNG!
   NẾU VI PHẠM → HÌNH VẼ SAI HOÀN TOÀN!
   
   LƯU Ý VỀ GÓC:
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
   (equal-distance A B 1.0)   ← 🔥 BẮT BUỘC: Constraint cho B để tiếp tuyến ổn định!
   (segment A B)
   (tangent A (circle O) AB)
   (segment O A)              ← 🔥🔥 BẮT BUỘC: Vẽ bán kính NGAY SAU tangent - KHÔNG ĐƯỢC CHEN GÌ!
   (define C point)
   (on-circle C O)            ← C trên đường tròn
   (define D point)
   (on-circle D O)            ← D trên đường tròn
   (segment C D)              ← Dây CD
   (parallel (segment A B) (segment C D))
   (segment A C)              ← BẮT BUỘC: Vẽ segment AC để thấy độ dài AC
   (segment A D)              ← BẮT BUỘC: Vẽ segment AD để thấy độ dài AD
   
   LƯU Ý QUAN TRỌNG:
   • Khi đề yêu cầu chứng minh AC = AD → CẦN vẽ segment AC và AD
   • Khi có tiếp tuyến tại A → CẦN vẽ bán kính OA NGAY SAU tangent
   • B phải có constraint (equal-distance A B 1.0) để tiếp tuyến ổn định
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
   
   LỖI 6: Thiếu constraint khi đề bài mô tả quan hệ hình học CỰC KỲ QUAN TRỌNG
   • ĐÚNG: "Chứng minh AB = AC" → THÊM (equal-distance A B A C) để vẽ AB = AC
   • ĐÚNG: "Chứng minh AB ⟂ CD" → THÊM (perpendicular (segment A B) (segment C D))
   • ĐÚNG: "Chứng minh góc AOC = góc BAC" → THÊM (angle-equal A O C B A C)
   • ĐÚNG: "Tính góc A" (nếu đề cho kết quả 60°) → THÊM (angle-measure B A C 60)
   • ĐÚNG: "Cho góc A = 60°" → THÊM (angle-measure B A C 60)
   • NGUYÊN TẮC: VẼ TẤT CẢ quan hệ hình học được mô tả, không phân biệt "cho" hay "chứng minh"
   
   VÍ DỤ CHI TIẾT:
   ĐÚNG: "Chứng minh rằng góc AOC = góc BAC"
   (angle-equal A O C B A C)  ← ĐÚNG - vẽ 2 góc bằng nhau!
   
   ĐÚNG: "Chứng minh AB = AC"
   (equal-distance A B A C)  ← ĐÚNG - vẽ AB và AC bằng nhau!
   
   NGUYÊN TẮC MỚI:
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
   
   ⚠️ LƯU Ý: TÊN ĐIỂM TRONG CÁC VÍ DỤ (A, O, C, B...) CHỈ LÀ VÍ DỤ!
   Đề dùng tên khác → Áp dụng quy tắc giống hệt nhưng thay tên!
   
   ĐỊNH NGHĨA TỔNG QUÁT:
   Góc ở tâm là góc có đỉnh trùng với tâm đường tròn.
   
   NHẬN BIẾT: Nếu đỉnh góc (chữ ở giữa) là tên tâm đường tròn → đó là GÓC Ở TÂM
   • Ví dụ: ∠AOC (đỉnh O là tâm) → góc ở tâm
   • Ví dụ: ∠ABC (đỉnh B không phải tâm) → không phải góc ở tâm
   
   QUY TẮC VẼ GÓC Ở TÂM:
   1. Kiểm tra các điểm đã được nối với tâm O chưa
   2. CHỈ vẽ bán kính CHƯA CÓ (tránh trùng lặp)
   3. Khai báo góc: (angle-measure A O C 50)
   
   TRƯỜNG HỢP ĐẶC BIỆT - KHI CÓ ĐƯỜNG KÍNH:
   • Nếu AB là đường kính → O đã nối với A và B
   • KHÔNG vẽ lại (segment O A) hay (segment O B) → bị trùng!
   • CHỈ vẽ bán kính mới đến điểm chưa nối
   
   VÍ DỤ CỤ THỂ 1: "AB là đường kính, C trên đường tròn, góc AOC = 50°"
   (diameter A B O)     ← AB là đường kính → OA, OB đã có
   (define C point)
   (on-circle C O)
   (segment O C)        ← CHỈ vẽ bán kính mới OC
   (angle-measure A O C 50)
   
   VÍ DỤ CỤ THỂ 2: "Dây AB, góc AOB = 120°"
   (segment A B)        ← Dây thường
   (segment O A)        ← CẦN vẽ bán kính OA
   (segment O B)        ← CẦN vẽ bán kính OB
   (angle-measure A O B 120)

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
   
   QUY TẮC MIDPOINT/PROJECTION:
   • TRƯỚC KHI dùng (midpoint B C) hoặc (projection A (segment B C))
   • Segment B C phải ĐÃ TỒN TẠI (từ triangle, quadrilateral, hoặc định nghĩa riêng)
   • Nếu B C CHƯA được nối → thêm (segment B C) trước midpoint/projection
   • Nếu B C ĐÃ CÓ SẴN (ví dụ từ triangle/quad) → KHÔNG thêm nữa (sẽ bị trùng!)
   • Ví dụ: (triangle (A B C)) → 3 cạnh AB, BC, CA đã có → (define M point (midpoint B C)) OK luôn
   • Ví dụ: B, C riêng lẻ → (segment B C) + (define M point (midpoint B C))

═══ VÍ DỤ ═══

⚠️ LƯU Ý QUAN TRỌNG VỀ VÍ DỤ:
Các ví dụ dưới đây dùng tên điểm CỤ THỂ (A, B, C, M, O, D...) CHỈ ĐỂ MINH HỌA!
Khi làm bài, BẠN PHẢI ĐỌC ĐỀ và DÙNG ĐÚNG TÊN ĐIỂM TRONG ĐỀ BÀI!
- Đề gọi "I" thì dùng I, không phải M
- Đề gọi "EF" thì dùng E, F, không phải A, B
- NGUYÊN TẮC: Hiểu logic, áp dụng với tên điểm thật trong đề!

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

⚠️ LƯU Ý: Tên điểm trong mẫu (A, B, C, M, N, O...) CHỈ LÀ VÍ DỤ! 
Đọc đề và thay bằng tên điểm thực tế. Ví dụ: Đề gọi "P, Q" thì dùng P, Q thay vì A, B

🔥🔥QUY TẮC VÀNG - TUYỆT ĐỐI KHÔNG ĐƯỢC VI PHẠM 
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
(segment O A)               ← BẮT BUỘC: VẼ BÁN KÍNH từ tâm đến tiếp điểm NGAY SAU TANGENT!
KHÔNG DÙNG (on-segment A B) - chỉ có 2 điểm!

DẠNG 2: "Từ A ngoài (O) kẻ tiếp tuyến AB và AC (B, C tiếp điểm)" - HAI TIẾP TUYẾN
(define O point)
(circle O)
(define A point)
(distance O A 1.5)          ← A nằm ngoài - BẮT BUỘC
(define B point)
(on-circle B O)
(segment A B)
(tangent B (circle O) AB)   ← Tiếp tuyến thứ nhất
(segment O B)               ← BÁN KÍNH OB - BẮT BUỘC NGAY SAU TANGENT!
(define C point)
(on-circle C O)
(segment A C)
(tangent C (circle O) AC)   ← Tiếp tuyến thứ hai
(segment O C)               ← BÁN KÍNH OC - BẮT BUỘC NGAY SAU TANGENT!
CẢ HAI BÁN KÍNH ĐỀU PHẢI VẼ!
KHÔNG DÙNG (on-segment ...) - B và C đều là endpoint!

DẠNG 3: "Đường kính MN, vẽ tiếp tuyến tại M" - TIẾP ĐIỂM TRÙNG ĐIỂM ĐƯỜNG KÍNH
(define O point)
(circle O)
(define M point)
(define N point)
(diameter M N O)          ← Đường kính MN
(segment M N)             ← BẮT BUỘC: Vẽ đường kính!
(define A point)          ← Điểm trên tiếp tuyến (KHÔNG define M lại!)
(segment M A)             ← MA là tiếp tuyến - M là endpoint
(tangent M (circle O) MA) ← Tiếp tuyến MA tại M
⚠️⚠️ KHÔNG VẼ (segment O M) - M đã trên đường kính!
⚠️⚠️ Bán kính OM ĐÃ TỒN TẠI trong (diameter M N O), vẽ lại = TRÙNG LẶP!

🔥🔥 QUAN TRỌNG - NẾU ĐỀ NÓI "TIẾP TUYẾN VUÔNG GÓC VỚI MN":
• Đề: "Chứng minh tiếp tuyến tại M vuông góc với MN"
• Đề: "Tiếp tuyến tại M vuông góc với đường kính"
• Đề: "Vẽ tiếp tuyến tại M. Chứng minh tiếp tuyến ⊥ MN"

⚠️ HIỂU RÕ VẤN ĐỀ:
- Đề chỉ nói "tiếp tuyến tại M" → chưa có tên cụ thể cho tiếp tuyến
- BẠN PHẢI TỰ ĐẶT TÊN cho điểm trên tiếp tuyến (ví dụ: A, P, Q, T...)
- Nếu đặt tên là P → tiếp tuyến sẽ tên là MP
- Khi đề nói "tiếp tuyến vuông góc MN" → MP (tiếp tuyến) vuông góc MN

→ DÙNG: (perpendicular (segment M P) (segment M N))
  * M P là tiếp tuyến (P là điểm TỰ ĐẶT TÊN)
→ KHÔNG DÙNG: (perpendicular (segment O M) (segment M N)) ← SAI!
  * O M là bán kính, KHÔNG PHẢI tiếp tuyến!

VÍ DỤ ĐẦY ĐỦ: "Đường kính MN, vẽ tiếp tuyến tại M. Chứng minh tiếp tuyến ⊥ MN"
(define O point)
(circle O)
(define M point)
(define N point)
(diameter M N O)
(segment M N)
(define A point)              ← TỰ ĐẶT TÊN điểm trên tiếp tuyến (có thể là A, P, Q...)
(segment M A)                 ← MA là tiếp tuyến
(tangent M (circle O) MA)     ← Khai báo MA là tiếp tuyến tại M
(perpendicular (segment M A) (segment M N))  ← ✅ ĐÚNG: MA (tiếp tuyến) ⊥ MN
KHÔNG DÙNG: (perpendicular (segment O M) (segment M N))  ← ❌ SAI! OM là bán kính!

GHI NHỚ:
• M là ENDPOINT của MA → KHÔNG CẦN on-segment
• M đã trên đường kính → KHÔNG CẦN (segment O M)
• "Tiếp tuyến vuông góc" → dùng TIẾP TUYẾN (MA), không phải bán kính (OM)
• Nếu đề nói "tiếp tuyến AB đi qua M" (M GIỮA A và B) → CẦN (segment A B) + (on-segment M A B)

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
(segment O A)               ← BÁN KÍNH từ tâm đến tiếp điểm - BẮT BUỘC NGAY SAU TANGENT!
(define C point)
(on-circle C O)             ← C là đầu kia của dây - BẮT BUỘC on-circle
(segment A C)               ← Vẽ dây AC
(segment O C)               ← BÁN KÍNH OC cần vẽ nếu đề nhắc (ví dụ: góc OCA, tam giác OAC,...)

═══ OUTPUT FORMAT ═══
TUYỆT ĐỐI QUAN TRỌNG:

1. CHỈ trả về JSON: [{"instruction": "...", "answer": "DSL với \\n"}]

2. Field "instruction" PHẢI là bài toán TIẾNG VIỆT gốc, KHÔNG được thay đổi!
   SAI: "Convert the geometry problem to GMBL"
   ĐÚNG: "Cho đường tròn (O) có đường kính MN..."

3. Field "answer" CHỈ chứa DSL thuần túy:
   • KHÔNG có comment #
   • KHÔNG có giải thích
   • CHỈ có DSL với \\n

4. KHÔNG markdown, KHÔNG giải thích bên ngoài JSON

CHECKLIST TRƯỚC KHI OUTPUT - KIỂM TRA KỸ DSL:

⚠️⚠️ LƯU Ý VỀ TÊN ĐIỂM TRONG CHECKLIST:
TẤT CẢ tên điểm trong checklist (M, H, C, D, A, B...) CHỈ LÀ VÍ DỤ MINH HỌA!
Áp dụng quy tắc cho BẤT KỲ TÊN ĐIỂM NÀO trong đề bài của bạn.
Ví dụ: "Kiểm tra M" = kiểm tra điểm bất kỳ (đề gọi I thì kiểm tra I, đề gọi P thì kiểm tra P)

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

11b. ✓ HÌNH CHIẾU = TRUNG ĐIỂM - TUYỆT ĐỐI KHÔNG DEFINE 2 LẦN!
    → "Kẻ OH ⊥ AC (H ∈ AC). Chứng minh H là trung điểm AC"
    → H ĐÃ LÀ hình chiếu → H CHÍNH LÀ trung điểm (do tính chất hình học)
    → CHỈ DÙNG: (define H point (projection O (segment A C)))
    → KHÔNG ĐƯỢC thêm: (define M point (midpoint A C)) ← LỖI trùng điểm!
    → Khi đề yêu cầu "chứng minh H là trung điểm" → H đã được xác định bởi projection, không cần midpoint riêng

12. ✓ Tiếp tuyến: Kiểm tra format (tangent M (circle O) AB)
   → M là tiếp điểm, O là tâm, AB là chuỗi 2 ký tự tạo đường thẳng
   → 🔥 PHẢI có (segment O M) Ở DÒNG NGAY SAU (tangent ...) - bán kính từ tâm đến tiếp điểm!
   → ⚠️ Ngoại lệ duy nhất: M đã nằm trên đường kính (diameter M N O) → BỎ QUA (segment O M)

13. ✓ Hình chiếu: Kiểm tra segment được vẽ từ ĐIỂM GỐC đến HÌNH CHIẾU
   → (projection O (segment A C)) → PHẢI có (segment O H) - TỪ O ĐẾN H!
   → KHÔNG ĐƯỢC (segment A H) - SAI!

13b. ✓ HÌNH CHIẾU - TUYỆT ĐỐI KHÔNG VẼ SEGMENT THỪA!
   → Khi H là hình chiếu lên AC: H ĐÃ TỰ ĐỘNG NẰM TRÊN AC
   → AC đã được vẽ bằng (segment A C)
   → TUYỆT ĐỐI KHÔNG vẽ thêm (segment A H) hoặc (segment H C) - VẼ ĐÈ!
   → CHỈ VẼ: (segment A C) + (define H point (projection O (segment A C))) + (segment O H)
   → KHÔNG BAO GIỜ VẼ THÊM: (segment A H) (segment H C) 

13b2. ✓ 🔥 HÌNH CHIẾU - TUYỆT ĐỐI KHÔNG THÊM PERPENDICULAR!
   → Khi dùng (define H point (projection O (segment A C)))
   → Projection ĐÃ TỰ ĐỘNG tạo constraint vuông góc OH ⊥ AC
   → TUYỆT ĐỐI KHÔNG thêm: (perpendicular (segment O H) (segment A C))
   → THÊM perpendicular = THỪA = GÂY XUNG ĐỘT!

13c. ✓ 🔥 ĐƯỜNG TRUNG TRỰC - TUYỆT ĐỐI KHÔNG VẼ SEGMENT THỪA!
   → Khi chứng minh "AO là đường trung trực BC": M tự động nằm trên AO
   → CHỈ VẼ 3 THÀNH PHẦN: (segment B C) + (segment A O) + (perpendicular ...)
   → TUYỆT ĐỐI KHÔNG define M và KHÔNG vẽ (segment A M) - VẼ ĐÈ!
   → Nếu đề KHÔNG yêu cầu dùng M → CHỈ VẼ 3 dòng trên, KHÔNG THÊM GÌ NỮA!
   → KHÔNG BAO GIỜ VẼ: (segment A M) (segment M O) 

13c2. ✓ 🔥 ĐƯỜNG TRUNG TRỰC + ĐỀ NHẮC - DÙNG INTER-LL!
   → Nếu đề BÀI NÓI "M là trung điểm BC" TRONG ĐỀ → PHẢI define M
   → PHẢI DÙNG: (define M point (inter-ll A O B C))
   → KHÔNG ĐƯỢC DÙNG: (define M point (midpoint B C)) ← SAI! Chấm sẽ lệch!
   → Lý do: inter-ll cho chấm M nằm CHÍNH XÁC tại giao điểm AO và BC

14. ✓ Khi có yêu cầu chứng minh về độ dài: PHẢI vẽ các segment tương ứng
   → "Chứng minh AC = AD" → PHẢI có (segment A C) và (segment A D)
   → "Chứng minh AB = CD" → PHẢI có (segment A B) và (segment C D)

15. ✓ QUY TẮC VÀNG - VẼ SEGMENT KHI ĐỀ NHẮC ĐẾN:
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

16b. ✓ 🔥 ĐƯỜNG KÍNH + TIẾP TUYẾN - KHÔNG VẼ BÁN KÍNH!
   → Nếu có (diameter M N O) VÀ (tangent M (circle O) MA)
   → KHÔNG ĐƯỢC có (segment O M) - M đã trên đường kính!
   → Bán kính OM đã tồn tại, vẽ lại = trùng lặp!

16c. ✓ 🔥 PERPENDICULAR VỚI TIẾP TUYẾN - PHẢI DÙNG ĐÚNG SEGMENT!
   → Đề: "tiếp tuyến tại M vuông góc với MN"
   → Đề: "Vẽ tiếp tuyến tại M. Chứng minh tiếp tuyến vuông góc đường kính"
   → NẾU đề chỉ nói "tiếp tuyến tại M" (không chỉ rõ tên):
      * BẠN PHẢI TỰ ĐẶT TÊN cho điểm trên tiếp tuyến (A, P, Q...)
      * Nếu đặt tên A → tiếp tuyến = MA
   → PHẢI DÙNG: (perpendicular (segment M A) (segment M N))
      * M A là tiếp tuyến (A là điểm TỰ ĐẶT TÊN)
   → KHÔNG DÙNG: (perpendicular (segment O M) (segment M N))
      * O M là bán kính, KHÔNG PHẢI tiếp tuyến!
   → KIỂM TRA: Tìm "tiếp tuyến" + "vuông góc" trong đề → dùng segment tiếp tuyến, không phải bán kính!

17. ✓ Không có comment # trong DSL output
   → Chỉ có DSL thuần túy với \n

18. ✓ Field "instruction" là bài toán gốc tiếng Việt, KHÔNG được thay đổi
   → Copy nguyên văn đề bài, không dịch sang tiếng Anh

19. ✓ Kiểm tra lại lần cuối: Có điểm nào được define 2 lần không?

20. ✓ 🔥 TIẾP TUYẾN - KIỂM TRA 2 ĐIỀU QUAN TRỌNG:
   a) SAU MỌI (tangent ...) → KIỂM TRA CÓ (segment O <tiếp-điểm>) NGAY SAU KHÔNG?
      → Nếu KHÔNG CÓ và tiếp điểm KHÔNG nằm trên đường kính → THÊM NGAY!
      → KHÔNG ĐƯỢC chen bất kỳ lệnh nào giữa tangent và segment O X!
      → Đây là lỗi PHỔ BIẾN NHẤT - hình sẽ VẼ SAI nếu sai thứ tự!
   
   b) ĐIỂM THỨ 2 TRÊN TIẾP TUYẾN (B) → KIỂM TRA CÓ CONSTRAINT KHÔNG?
      → VÍ DỤ: Tiếp tuyến AB (A là tiếp điểm) → B là điểm thứ 2
      → Nếu đề KHÔNG nói gì về B → PHẢI THÊM (equal-distance A B 1.0)
      → B không constraint = B tự do = tiếp tuyến không ổn định!
      → Đây là lỗi NGHIÊM TRỌNG THỨ 2 - hình sẽ không ổn định!

═══ BƯỚC CUỐI CÙNG - XÁC MINH DSL ═══

TRƯỚC KHI OUTPUT, PHÂN TÍCH ĐỀ BÀI VỀ TIẾP TUYẾN:

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

Bước 4: 🔥🔥 KIỂM TRA CONSTRAINT CHO ĐIỂM THỨ 2 TRÊN TIẾP TUYẾN 🔥🔥
• Tìm tiếp tuyến AB (A là tiếp điểm) → B là điểm thứ 2
• ĐỌC ĐỀ: Đề có nói gì về B không?
   - Nếu KHÔNG nói gì (B tự do) → PHẢI THÊM (equal-distance A B 1.0)
   - Nếu có nói (ví dụ: "B nằm trên...", "AB = ...") → THEO ĐỀ
• 🚨 QUAN TRỌNG: B không constraint = tiếp tuyến không ổn định!
• VÍ DỤ ĐÚNG:
   (define B point)
   (equal-distance A B 1.0)  ← THÊM dòng này!
   (segment A B)
   (tangent A (circle O) AB)

Bước 5: 🔥🔥 KIỂM TRA THỨ TỰ (segment O <tiếp-điểm>) 🔥🔥
• Tìm (tangent X (circle O) ...) trong DSL
• NGAY DÒNG TIẾP THEO phải là (segment O X)
• KHÔNG ĐƯỢC có BẤT KỲ lệnh nào chen giữa!
• VÍ DỤ:
   ❌ SAI:
   (tangent A (circle O) AB)
   (define C point)          ← SAI! Không được chen!
   (segment O A)             ← Quá muộn!
   
   ✅ ĐÚNG:
   (tangent A (circle O) AB)
   (segment O A)             ← NGAY DÒNG NÀY!
   (define C point)          ← Các điểm khác sau đó

Bước 6: Kiểm tra DSL
• Tìm (on-segment A B) - chỉ 2 điểm → XÓA ngay!
• Tìm (on-segment M A M) - điểm trùng → XÓA ngay!
• Đếm define cho mỗi điểm → nếu > 1 → XÓA các lần define thừa

TRƯỚC KHI OUTPUT, KIỂM TRA HÌNH CHIẾU VÀ VUÔNG GÓC:

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

TRƯỚC KHI OUTPUT, KIỂM tra SEGMENT CẦN THIẾT:

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

TRƯỚC KHI OUTPUT, KIỂM TRA ĐƯỜNG KÍNH VÀ SEGMENT:

Bước 1: Tìm tất cả quan hệ hình học được mô tả
• "Chứng minh AC = AD" → Vẽ AC = AD → Cần (equal-distance A C A D)
• "Chứng minh AB ⊥ CD" → Vẽ AB ⊥ CD → Cần (perpendicular (segment A B) (segment C D))
• "Chứng minh H là trung điểm AC" → Vẽ H ở giữa AC → Cần (define H point (midpoint A C))

Bước 2: Kiểm tra các segment cần vẽ
• "AC = AD" → CẦN (segment A C) và (segment A D)
• "AB ⊥ CD" → CẦN (segment A B) và (segment C D)
• "AB ⊥ CD" → CẦN (segment A B) và (segment C D) - KHÔNG thêm perpendicular!
• "H là trung điểm" → CẦN segment trước midpoint

TRƯỚC KHI OUTPUT, KIỂM TRA ĐƯỜNG KÍNH VÀ SEGMENT:

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

Bước 3: 🔥🔥🔥 Kiểm tra bán kính cho tiếp tuyến - CỰC KỲ QUAN TRỌNG! 🔥🔥🔥

⚠️ ĐÂY LÀ LỖI THƯỜNG GẶP NHẤT - PHẢI KIỂM TRA KỸ!

THUẬT TOÁN KIỂM TRA:
1. Tìm tất cả (tangent X (circle O) ...) trong DSL
2. Với MỖI tangent tìm được:
   a) Kiểm tra DÒNG NGAY SAU có (segment O X) không?
   b) NẾU CÓ → ✅ Đúng, bỏ qua
   c) NẾU KHÔNG CÓ → Chuyển sang bước 3
3. Kiểm tra X có nằm trên đường kính không?
   - Tìm trong DSL: (diameter X ... O) hoặc (diameter ... X O)
   - NẾU TÌM THẤY → X trên đường kính → KHÔNG cần thêm (segment O X)
   - NẾU KHÔNG TÌM THẤY → X là tiếp điểm tự do → ❌ LỖI! PHẢI SỬA NGAY!
4. Sửa lỗi: CHÈN (segment O X) vào NGAY SAU dòng (tangent X ...)

VÍ DỤ CỤ THỂ:

❌ LỖI 1 - Thiếu bán kính hoàn toàn:
(tangent A (circle O) AB)
(define C point)              ← ❌ Không có (segment O A) ở đây!
(on-circle C O)
(segment A C)
→ SỬA: Chèn (segment O A) ngay sau dòng tangent:
(tangent A (circle O) AB)
(segment O A)                 ← ✅ Thêm vào đây!
(define C point)

❌ LỖI 2 - Bán kính ở sai vị trí:
(tangent A (circle O) AB)
(define C point)
(on-circle C O)
(segment A C)
(segment O A)                 ← ❌ Quá muộn! Sai vị trí!
→ SỬA: Di chuyển (segment O A) lên ngay sau tangent:
(tangent A (circle O) AB)
(segment O A)                 ← ✅ Đúng vị trí!
(define C point)
(on-circle C O)
(segment A C)

✅ ĐÚNG - Không cần sửa:
(tangent B (circle O) AB)
(segment O B)                 ← ✅ Ngay dòng sau tangent!
(define C point)

✅ NGOẠI LỆ - Không cần sửa vì M trên đường kính:
(diameter M N O)
(segment M N)
(tangent M (circle O) MA)     ← M đã trên đường kính → không cần (segment O M)
(define A point)

🔥 NHẮC LẠI: (segment O X) PHẢI Ở DÒNG NGAY SAU (tangent X ...)
             KHÔNG CÓ BẤT KỲ DÒNG NÀO CHEN GIỮA!

Bước 4: 🔥🔥🔥 Kiểm tra perpendicular với tiếp tuyến - CỰC KỲ QUAN TRỌNG! 🔥🔥🔥

⚠️ LỖI THƯỜNG GẶP: Dùng bán kính thay vì tiếp tuyến trong perpendicular!

THUẬT TOÁN KIỂM TRA:
1. Đọc đề bài: Tìm "tiếp tuyến" + "vuông góc"
   • "Chứng minh tiếp tuyến tại M vuông góc với MN"
   • "Vẽ tiếp tuyến tại M. Chứng minh tiếp tuyến ⊥ MN"
   • "tiếp tuyến vuông góc đường kính"

2. Xác định tiếp tuyến là đoạn nào:
   • NẾU đề chỉ nói "tiếp tuyến tại M" mà KHÔNG CHỈ RÕ tên điểm khác:
     → BẠN PHẢI TỰ ĐẶT TÊN cho điểm trên tiếp tuyến (A, P, Q, T...)
     → Nếu tự đặt tên là P → tiếp tuyến là MP
     → Khi đề nói "tiếp tuyến vuông góc MN" → MP vuông góc MN
   • NẾU đề nói rõ: "tiếp tuyến MA" hoặc "AB là tiếp tuyến tại A":
     → Tiếp tuyến đã có tên sẵn (MA hoặc AB)
   
3. Kiểm tra perpendicular trong DSL:
   • PHẢI DÙNG segment của TIẾP TUYẾN:
     - (segment M P) nếu P là điểm tự đặt tên
     - (segment M A) nếu A là điểm tự đặt tên
     - (segment A B) nếu đề nói "AB là tiếp tuyến"
   • KHÔNG ĐƯỢC dùng segment của BÁN KÍNH:
     - (segment O M) ← BÁN KÍNH, không phải tiếp tuyến!
     - (segment O A) ← BÁN KÍNH, không phải tiếp tuyến!

VÍ DỤ CỤ THỂ:

❌ LỖI - Dùng bán kính thay vì tiếp tuyến:
Đề: "Đường kính MN, vẽ tiếp tuyến tại M. Chứng minh tiếp tuyến ⊥ MN"
DSL SAI:
(define A point)              ← Tự đặt tên điểm A
(segment M A)
(tangent M (circle O) MA)
(perpendicular (segment O M) (segment M N))  ← ❌ SAI! O M là bán kính, không phải tiếp tuyến MA!

DSL ĐÚNG:
(define A point)              ← Tự đặt tên điểm A (hoặc P, Q...)
(segment M A)                 ← MA là tiếp tuyến
(tangent M (circle O) MA)
(perpendicular (segment M A) (segment M N))  ← ✅ ĐÚNG! M A (tiếp tuyến) ⊥ MN

GIẢI THÍCH:
• Đề chỉ nói "tiếp tuyến tại M" → không có tên cụ thể
• Bạn tự đặt tên điểm A trên tiếp tuyến → tiếp tuyến = MA
• "Tiếp tuyến vuông góc MN" = MA vuông góc MN
• KHÔNG PHẢI OM (bán kính) vuông góc MN!

✅ VÍ DỤ 2 - Tiếp tuyến AB vuông góc bán kính OA:
Đề: "AB là tiếp tuyến tại A. Chứng minh AB ⊥ OA"
DSL ĐÚNG:
(tangent A (circle O) AB)
(segment O A)                                ← Bán kính OA
(perpendicular (segment A B) (segment O A))  ← ✅ ĐÚNG! A B (tiếp tuyến) ⊥ O A (bán kính)
hoặc:
(perpendicular (segment O A) (segment A B))  ← ✅ Cũng đúng! O A ⊥ A B

LƯU Ý: Trong trường hợp này:
• AB là tiếp tuyến
• OA là bán kính đến tiếp điểm A
• Tiếp tuyến luôn vuông góc bán kính tại tiếp điểm → AB ⊥ OA là đúng
• Dùng (segment A B) cho tiếp tuyến, (segment O A) cho bán kính

🔥 QUY TẮC: "Tiếp tuyến vuông góc" → DÙNG segment tiếp tuyến, KHÔNG phải bán kính!

Bước 5: 🔥🔥🔥 Kiểm tra constraint cho điểm thứ 2 trên tiếp tuyến - CỰC KỲ QUAN TRỌNG! 🔥🔥🔥

⚠️ LỖI THƯỜNG GẶP: Điểm thứ 2 trên tiếp tuyến không có constraint → tiếp tuyến không ổn định!

THUẬT TOÁN KIỂM TRA:
1. Tìm tất cả tiếp tuyến trong DSL:
   • (tangent A (circle O) AB) → A là tiếp điểm, B là điểm thứ 2
   • (tangent M (circle O) MA) → M là tiếp điểm, A là điểm thứ 2

2. Với MỖI tiếp tuyến, xác định điểm thứ 2:
   • Tiếp tuyến AB, A là tiếp điểm → B là điểm thứ 2
   • Tiếp tuyến MA, M là tiếp điểm → A là điểm thứ 2

3. Kiểm tra DSL - Điểm thứ 2 có constraint không?
   • Tìm các constraint liên quan đến B:
     - (distance A B ...) ✓
     - (equal-distance A B ...) ✓
     - (on-circle B O) - NẾU B trên đường tròn khác ✓
     - (on-segment B ...) - NẾU B nằm trên đoạn thẳng nào đó ✓
   • NẾU KHÔNG TÌM THẤY BẤT KỲ constraint nào → ❌ LỖI!

4. ĐỌC ĐỀ BÀI - Đề có nói gì về điểm thứ 2 không?
   • NẾU đề nói: "B nằm trên...", "AB = ...", "B là..." → THEO ĐỀ
   • NẾU đề KHÔNG nói gì về B (B tự do) → PHẢI THÊM constraint!

5. Sửa lỗi: CHÈN (equal-distance X Y 1.0) ngay sau (define Y point)

VÍ DỤ CỤ THỂ:

❌ LỖI - B không có constraint:
(define A point)
(on-circle A O)              ← A là tiếp điểm, có constraint (trên đường tròn)
(define B point)             ← ❌ B không có constraint gì cả!
(segment A B)
(tangent A (circle O) AB)
→ HẬU QUẢ: B có thể ở BẤT KỲ ĐÂU → AB không ổn định!

✅ SỬA - Thêm constraint cho B:
(define A point)
(on-circle A O)              ← A là tiếp điểm
(define B point)
(equal-distance A B 1.0)     ← ✅ Cho AB có độ dài cố định 1.0!
(segment A B)
(tangent A (circle O) AB)
→ KẾT QUẢ: B cách A một khoảng cố định → AB ổn định!

✅ VÍ DỤ 2 - Đề có nói về B:
Đề: "Từ A kẻ tiếp tuyến AB, B nằm trên đường thẳng d"
(define A point)
(on-circle A O)
(define B point)
(on-line B d)                ← ✅ B có constraint (nằm trên đường thẳng d)
(segment A B)
(tangent A (circle O) AB)
→ KẾT QUẢ: B có constraint rồi, không cần thêm!

🔥 QUY TẮC: Mỗi điểm PHẢI có ÍT NHẤT 1 constraint!
           Điểm không constraint = điểm tự do = không ổn định!

TRƯỚC KHI OUTPUT, KIỂM TRA TỪNG DÒNG DSL:

1. Đếm số lần define cho MỖI điểm → Phải = 1
   • Tìm: (define A point) - có bao nhiêu lần?
   • Tìm: (define M point) - có bao nhiêu lần?
   • Nếu > 1 → XÓA các lần define thừa

2. Tìm tất cả (on-segment ...) → Đếm số điểm
   • (on-segment M A B) - 3 điểm ✓
   • (on-segment A B) - 2 điểm ✗ → XÓA (A là endpoint, không cần on-segment)
   • (on-segment M A M) - điểm trùng ✗ → XÓA
   • (on-segment A A B) - điểm trùng ✗ → XÓA
   
   QUY TẮC ĐẶC BIỆT CHO TIẾP TUYẾN:
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

6. 🔥 Tìm (projection ...) → Kiểm tra KHÔNG có segment THỪA!
   • Tìm: (define H point (projection O (segment A C)))
   • Kiểm tra: Có (segment A H) hoặc (segment H C) KHÔNG?
   • Nếu CÓ → XÓA NGAY! (H đã nằm trên AC, vẽ thêm là VẼ ĐÈ)
   • CHỈ GIỮ LẠI: (segment O H) - từ điểm gốc đến hình chiếu

6b. 🔥 Tìm (projection ...) + "chứng minh vuông góc" → PHẢI CÓ perpendicular!
   • Tìm: (define H point (projection O (segment A C)))
   • Xem instruction: Có "chứng minh ... vuông góc" hoặc "chứng minh H là trung điểm"?
   • Kiểm tra DSL: Có (perpendicular (segment O H) (segment A C)) KHÔNG?
   • Nếu KHÔNG → THÊM NGAY sau (segment O H)!

7. 🔥 Tìm "đường trung trực" → KIỂM TRA ĐỀ CÓ NHẮC TÊN ĐIỂM GIAO KHÔNG?
   • Bước 1 - 🔍 TÌM TRONG ĐỀ BÀI (instruction) - TÌM BẤT KỲ TÊN ĐIỂM NÀO:
     - Tìm các cụm: "Gọi [X]", "[X] là giao điểm", "điểm [X]", "Lấy [X]", "[X] thỏa mãn"
     - VÍ DỤ: "Gọi M", "M là", "I là giao điểm", "điểm H", "N nằm trên", "Lấy P"
     - [X] có thể là: M, I, H, N, P, K, Q, D, E, F... BẤT KỲ TÊN NÀO!
   
   • Bước 2a - NẾU ĐỀ KHÔNG NHẮC TÊN ĐIỂM GIAO NÀO:
     - ĐỀ CHỈ NÓI: "AO là đường trung trực của BC" (không nhắc tên điểm)
     - CHỈ GIỮ LẠI 3 DÒNG: (segment B C) + (segment A O) + (perpendicular ...)
     - TUYỆT ĐỐI KHÔNG define điểm nào, KHÔNG vẽ segment thừa!
   
   • Bước 2b - NẾU ĐỀ CÓ NHẮC TÊN ĐIỂM (M, I, H, N...):
     - Đề nói: "Gọi M là giao điểm..." HOẶC "I là trung điểm BC" HOẶC "Điểm H..."
     - PHẢI DEFINE điểm đó bằng: (define [X] point (inter-ll A O B C))
     - VÍ DỤ: (define M...) hoặc (define I...) hoặc (define H...) - tùy đề gọi tên gì
     - VẼ ĐẦY ĐỦ: (segment B C) + (segment A O) + (perpendicular ...) + (define [X]...)

7b. 🔥 Tìm đường trung trực + điểm giao → PHẢI DÙNG inter-ll, KHÔNG DÙNG midpoint!
   • Kiểm tra DSL: Có (define [X] point (midpoint B C))? (X = tên điểm bất kỳ)
   • Nếu CÓ và đề bài nhắc [X] là giao điểm → ĐỔI THÀNH: (define [X] point (inter-ll A O B C))
   • VÍ DỤ: (define M...) hoặc (define I...) hoặc (define H...) - đều phải dùng inter-ll
   • Lý do: Chấm phải nằm CHÍNH XÁC tại giao điểm, dùng midpoint sẽ bị lệch khỏi đường AO!

═══ INPUT ═══
{{ extract }}

"""
