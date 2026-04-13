Diagram Rendering trong GeoUni
1. Mục tiêu
Diagram Rendering là giai đoạn chuyển cấu trúc hình học đã tối ưu tọa độ thành ảnh trực quan phục vụ người dùng và pipeline API.
Mục tiêu của thành phần này gồm:

Biểu diễn chính xác đối tượng hình học đã được optimizer giải ra.
Thể hiện rõ các ký hiệu hình học quan trọng: cạnh bằng nhau, góc vuông, góc bằng nhau, số đo góc, đường phụ.
Đảm bảo bố cục dễ đọc, cân tâm, không bị cắt hình.
Trả kết quả ảnh dưới dạng phù hợp cho web/API (PNG hoặc base64 PNG).
2. Vị trí trong pipeline tổng thể
Luồng render trong hệ thống:

LLM sinh DSL từ đề bài tự nhiên.
Optimizer giải ràng buộc và tạo Diagram object chứa toàn bộ thực thể hình học.
Renderer đọc Diagram object và vẽ ảnh bằng Matplotlib.
Ảnh được trả về qua API đồng bộ hoặc bất đồng bộ.
Trong luồng bất đồng bộ:

API enqueue task render.
Celery worker thực thi render.
Client poll trạng thái task để lấy ảnh kết quả.
3. Kiến trúc module liên quan
Các thành phần chính:

Renderer chính: matplotlib_renderer.py
Celery worker render task: tasks.py
API queue và status: tasks.py
Task queue service: queue.py
Cấu trúc dữ liệu Diagram: entities.py
4. Mô hình dữ liệu đầu vào cho renderer
Renderer nhận một Diagram object, trong đó các trường cốt lõi gồm:

points: map tên điểm -> tọa độ GeometricPoint
triangles: danh sách tam giác, có metadata về cạnh bằng nhau, góc vuông, góc bằng nhau
quadrilaterals: danh sách tứ giác, có type (square, rectangle, rhombus, general)
circles: tâm + thông tin bán kính và loại đường tròn
segments: các đoạn phụ trợ (thường nét đứt)
lines: các đường thẳng vô hạn
angle_bisectors, angle_equal_assertions, angle_measures
perpendiculars: metadata để vẽ marker vuông góc tại giao điểm
Thiết kế này tách logic hình học khỏi logic hiển thị, giúp renderer chỉ tập trung vào visualization.

5. Quy trình render chi tiết
Hàm render thực hiện theo thứ tự lớp vẽ:

Khởi tạo figure/axes.
Vẽ tam giác.
Vẽ tứ giác.
Vẽ đường tròn.
Vẽ segments phụ trợ.
Vẽ marker vuông góc.
Vẽ lines vô hạn.
Vẽ điểm và nhãn.
Vẽ ký hiệu góc (bisector, angle-equal, angle-measure).
Tính bounding box, cân tâm và zoom.
Xuất ảnh hoặc trả fig/ax.
Thứ tự này giúp lớp thông tin quan trọng không bị che khuất và ảnh cuối cùng dễ đọc.

6. Primitive rendering quan trọng
6.1 Tick marks cho cạnh bằng nhau
Renderer vẽ một hoặc nhiều gạch nhỏ trên cạnh để biểu diễn các nhóm cạnh bằng nhau.
Ý tưởng hình học:

Lấy trung điểm cạnh.
Tính vector pháp tuyến của cạnh.
Vẽ các đoạn ngắn vuông góc cạnh tại vùng trung điểm với khoảng cách đều nhau.
6.2 Ký hiệu góc vuông
Ký hiệu vuông góc được vẽ bằng một hình vuông nhỏ tại đỉnh góc, dựng theo hai vector chuẩn hóa đi ra từ đỉnh.

6.3 Cung góc và dấu bằng nhau
Góc bằng nhau được biểu diễn bằng cung tròn nhỏ tại đỉnh, có thể kèm dấu gạch trên cung để phân biệt với góc thường.

6.4 Số đo góc
Với angle measure:

Góc 90 độ: ưu tiên marker vuông, bỏ text số đo.
Góc khác 90 độ: vẽ cung xanh + text số đo ở vị trí giữa cung.
6.5 Giao điểm hai đoạn
Khi cần vẽ marker vuông góc cho hai đoạn cắt nhau, renderer tính giao điểm đoạn-đoạn bằng công thức tham số và chỉ vẽ marker nếu giao nằm trong cả hai đoạn.

7. Quy tắc hiển thị theo loại hình
Tam giác
Luôn vẽ 3 cạnh chính nét liền đen.
Có thể thêm:
Tick marks cho cạnh bằng nhau.
Marker góc vuông nếu metadata chỉ định.
Marker góc bằng nhau nếu có assertion hoặc metadata.
Tứ giác
Theo type:

Square: 4 góc vuông + 4 cạnh có cùng tick.
Rectangle: 4 góc vuông + 2 cặp cạnh đối diện với số tick khác nhau.
Rhombus: 4 cạnh cùng tick, không tự động thêm đường chéo trừ khi được khai báo.
General quadrilateral: chỉ vẽ đường bao.
Đường tròn
Dữ liệu circle có thể là:

Dạng dict chứa radius và type.
Dạng fallback bán kính số.
Màu theo type thường dùng:

incircle: xanh dương
circumcircle: xanh lá
positioned hoặc default: đen
Segment phụ trợ
Vẽ nét đứt, độ đậm thấp hơn đường chính để tránh rối.

Line vô hạn
Từ hai điểm mốc, renderer kéo dài line theo cả hai phía và thêm mũi tên để thể hiện tính vô hạn.

8. Quản lý bố cục và viewport
Để ảnh không bị lệch tâm hoặc quá nhỏ:

Tính bounding box từ toàn bộ points.
Mở rộng thêm theo bán kính các circles.
Lấy tâm thực của tất cả phần tử để đặt camera.
Dùng zoom factor có ngưỡng tối thiểu nhằm tránh hình quá tiny.
Tắt trục tọa độ để cho ảnh sạch.
Cơ chế này giúp output ổn định hơn trên nhiều đề bài khác nhau.

9. Xuất ảnh và tích hợp API
Xuất ảnh tại renderer
Hỗ trợ save file PNG.
Có thể chỉ trả fig, ax cho pipeline xử lý tiếp.
Bất đồng bộ qua Celery
API tạo task với dsl, epochs, n_tries, dpi.
Worker render và trả ảnh base64.
Endpoint status cung cấp trạng thái PENDING, STARTED, SUCCESS, FAILURE.
DPI được expose trong request model để điều chỉnh chất lượng ảnh theo nhu cầu.

10. Độ phức tạp và hiệu năng rendering
Chi phí render chủ yếu tỷ lệ với số lượng thực thể cần vẽ:

Points, segments, lines, circles
Số marker góc/cạnh
Mức độ annotation
Về thực tế, rendering nhẹ hơn optimization; bottleneck chính thường nằm ở bước solve ràng buộc, không phải vẽ.

11. Cơ chế robust và fallback
Renderer có các hành vi an toàn:

Nếu diagram rỗng, vẫn tạo figure placeholder thay vì crash.
Bỏ qua marker nếu vector quá ngắn (tránh chia cho 0).
Chỉ vẽ giao điểm vuông góc khi thật sự có giao trên đoạn.
Bỏ các điểm tên phụ có hậu tố aux trong lớp nhãn để giảm nhiễu thị giác.
12. Hạn chế hiện tại
Một số điểm cần lưu ý khi viết paper:

Quy tắc style hiện còn heuristic, chưa có style engine chuẩn hóa.
Với cấu hình hình học rất dày đặc, annotation có thể chồng lấn.
Việc chọn vị trí label hiện thiên về quy tắc cục bộ, chưa có global label optimization.
Luồng Celery đang gọi hàm render_dsl_to_image từ module rendering, cần đảm bảo module này tồn tại và đồng bộ với renderer thực tế.
13. Hướng cải thiện đề xuất
Các nâng cấp có giá trị nghiên cứu/sản phẩm:

Auto label placement bằng force-based hoặc simulated annealing.
Theme system cho ký hiệu hình học (màu, nét, cỡ chữ) theo profile.
Vector output chuẩn hóa SVG/PDF cho tài liệu học thuật.
Caching theo hash của DSL để tránh render lặp.
Rendering quality metrics: overlap rate, legibility score, marker clarity.
14. Đoạn paper-ready (có thể đưa vào Method)
Diagram rendering is implemented as a deterministic visualization stage that consumes optimized geometric states and produces publication-ready diagrams. The renderer draws geometric primitives in layered order (polygons, circles, auxiliary segments, lines, points, and semantic annotations) to preserve visual clarity. Symbolic constraints solved during optimization are translated into standard geometric markers, including equal-side ticks, right-angle squares, equal-angle arcs, and degree labels. A bounding-box-based viewport strategy, expanded by circle radii, ensures centered and stable outputs across problem types. For production inference, rendering is executed asynchronously through Celery workers and returned as base64 PNG, enabling responsive API behavior while keeping the optimization-rendering pipeline decoupled from request handling.

15. Tóm tắt
Diagram Rendering trong GeoUni là lớp chuyển đổi từ hình học số sang hình ảnh trực quan, đóng vai trò cầu nối cuối cùng giữa nghiệm tối ưu và trải nghiệm người dùng. Thành phần này kết hợp:

Quy tắc hình học có ngữ nghĩa
Kỹ thuật hiển thị ổn định
Tích hợp API bất đồng bộ
để đảm bảo ảnh đầu ra vừa đúng hình học, vừa rõ ràng khi sử dụng trong bài toán giáo dục.