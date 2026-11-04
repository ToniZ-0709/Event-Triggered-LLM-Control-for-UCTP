# Finalized Work Plan: Multi-Pipeline Timetabling Optimization

---

## 1. Thiết lập 5 Pipelines Thực nghiệm

Repo hiện có Baseline, Single-agent và Multi-agent. Random và Greedy là hai đối chứng dự kiến, chưa được triển khai.

| Pipeline | Cơ chế xác định WHERE và HOW | Cơ chế xác định WHEN |
| :--- | :--- | :--- |
| **Baseline** | Solver NSGA-III tự động vận hành kèm bộ repair mặc định | Không sử dụng controller |
| **Random** | Thuật toán chọn ngẫu nhiên vùng và toán tử hợp lệ (kèm tùy chọn `no_op`) | LLM chỉ định khoảng cách review tiếp theo |
| **Greedy** | Thuật toán heuristic chọn theo luật cố định (kèm tùy chọn `no_op`) | LLM chỉ định khoảng cách review tiếp theo |
| **Single-agent** | LLM phân tích toàn diện để chọn vùng, toán tử và budget trong 1 lượt gọi | LLM chỉ định khoảng cách review tiếp theo |
| **Multi-agent** | Phân vai tuần tự: Inspector chọn vùng $\rightarrow$ Planner chọn toán tử và budget | Planner chỉ định khoảng cách review tiếp theo |

### Nguyên tắc kỹ thuật chung:
- **Nền tảng chung:** Ba pipeline hiện tại dùng chung solver và repair. Theo kế hoạch, bốn controller sẽ dùng chung cơ chế sinh region, tập action và validator; Baseline không có controller.
- **Phần xử lý WHEN:** Hai pipeline Random và Greedy vẫn gọi LLM để quyết định nhịp độ review (`next_review_interval`). Cả hai dùng chung model, chung template prompt và dữ liệu đầu vào. Số lượt gọi API, số token và độ trễ được đo đạc độc lập.
- **Điều kiện kích hoạt:** Áp dụng cùng tập luật trigger (stagnation, low diversity, checkpoint định kỳ) và cùng khoảng giới hạn `next_review_interval` $\in [3, 20]$. Thời điểm kích hoạt thực tế giữa các pipeline có thể khác nhau do trạng thái tìm kiếm phân kỳ.

---

### 1.1. Cơ chế không can thiệp (`no_op`)
- `no_op` là một lựa chọn toàn cục của controller, không gắn riêng với bất kỳ vùng nào.
- Khi chọn `no_op`: `solution_id = region_id = boundary_id = null`, `operator_id = "no_op"`, `execution_budget = 0`. LLM vẫn tính toán và trả về `next_review_interval`.
- Nếu không tìm thấy toán tử hợp lệ hoặc đã cạn ngân sách đánh giá, controller chủ động trả về `no_op`. Trường hợp review bị hủy do hết thời gian chạy hoặc lỗi mạng API, hệ thống ghi nhận trạng thái lỗi riêng.
- Phân định rõ ràng: `no_op` do chính sách điều phối chọn khác hoàn toàn với `fallback no_op` do lỗi parse JSON hoặc lỗi API. Lỗi kỹ thuật không bao giờ được ghi nhận là một hành động hợp lệ.

---

### 1.2. Thiết kế Random Agent

| Bước | Quy tắc thực thi |
| :--- | :--- |
| **Không gian hành động** | Tạo tập hợp tất cả các cặp hợp lệ `(region, operator)` từ danh sách region được công bố và `available_actions.operators`, đồng thời bổ sung đúng 1 lựa chọn `no_op`. Toán tử `blocker_chain` chỉ xuất hiện khi region mở rộng được kích hoạt. |
| **Xác định WHERE / HOW** | Giả sử có $N$ cặp `(region, operator)` hợp lệ: lấy mẫu ngẫu nhiên phân phối đều trên $N + 1$ lựa chọn (xác suất chọn `no_op` là $1/(N+1)$). Region nào chứa nhiều toán tử hợp lệ sẽ có xác suất được chọn cao hơn. Nếu $N = 0$, mặc định chọn `no_op`. |
| **Ngân sách thực thi (Budget)** | Với các toán tử thông thường, gán budget cố định = $\min(10, \text{current\_max})$, còn với `no_op` thì gán budget = 0. Giá trị này được giữ cố định xuyên suốt chiến dịch để đảm bảo tính chuẩn hóa. |
| **Xác định WHEN** | Sau khi chốt action, gửi hai khối dữ liệu `global_search_state` và `recent_quality` cho LLM để chọn `next_review_interval` trong khoảng $[3, 20]$. |
| **Tính tái lập (Reproducibility)** | Sử dụng bộ sinh số ngẫu nhiên (RNG) độc lập với RNG của solver. Hệ thống ghi nhận seed, số cặp hợp lệ, xác suất chọn và kết quả hành động vào log để phục vụ kiểm toán. |

---

### 1.3. Thiết kế Greedy Agent

| Bước | Quy tắc thực thi |
| :--- | :--- |
| **Chọn Region (WHERE)** | Trong các region có toán tử hợp lệ, ưu tiên chọn cá thể nguồn theo thứ tự tăng dần của bộ khóa `(source_hard, source_internal_total, solution_id)`. Trong cá thể đó, chọn region có `hard_contribution` lớn nhất (nếu cá thể chưa khả thi) hoặc `soft_contribution` lớn nhất (nếu cá thể đã khả thi). Nếu hòa điểm, chọn theo thứ tự bảng chữ cái của `region_id`. |
| **Điều kiện chọn `no_op`** | Chọn `no_op` khi không tồn tại cặp region/toán tử hợp lệ, đã hết ngân sách đánh giá, hoặc toàn bộ điểm đóng góp phạt của các region trong cá thể đã chọn đều bằng 0 và không tồn tại `unlocalized_penalties`. |
| **Chọn Toán tử (HOW)** | Lựa chọn theo thứ tự ưu tiên nghiệp vụ:<br>1. `blocker_chain`: nếu có trong region mở rộng.<br>2. `distribution_repair`: nếu phát hiện ràng buộc phân phối bắt buộc đang bị vi phạm.<br>3. `destroy_repair`: nếu phát hiện xung đột phòng học.<br>4. Nếu không thỏa các điều kiện trên, chọn toán tử hợp lệ đầu tiên theo danh sách: `guided_move` $\rightarrow$ `move` $\rightarrow$ `swap` $\rightarrow$ `destroy_repair` $\rightarrow$ `distribution_repair` $\rightarrow$ `blocker_chain`. |
| **Ngân sách thực thi (Budget)** | Gán budget cố định = $\min(10, \text{current\_max})$, còn với `no_op` thì gán budget = 0. |
| **Xác định WHEN** | Tương tự Random Agent, LLM nhận hai khối tiến độ và lịch sử để quyết định `next_review_interval`. |

- **Bản chất điểm quy kết:** `hard_contribution` và `soft_contribution` là các ước lượng heuristic cục bộ, không thể cộng gộp để suy ra điểm vi phạm toàn cục. Thuật toán Greedy chỉ dùng các giá trị này để xếp hạng ưu tiên tương đối.
- **Xử lý dữ liệu khuyết:** Khi thiếu điểm số cá thể, ưu tiên cá thể có đủ cặp $(H, S)$ trước. Nếu cùng thiếu, thuật toán xét thứ tự `solution_id` rồi đến `region_id`. Dữ liệu thiếu không tự động dẫn đến `no_op`.
- **Minh bạch ngân sách:** Ngân sách trần mỗi lượt review là như nhau, nhưng budget thực tế của Single-agent và Multi-agent do LLM tự quyết định (có thể dao động từ 1 đến 24). Khi so sánh kết quả, bắt buộc phải đối chiếu kèm budget thực tế tiêu tốn, số lần review và chi phí token.

---

## 2. Tích hợp XAI (Giải thích Quyết định)

**Mục tiêu:** Cung cấp công cụ kiểm toán minh bạch cho từng lượt review: LLM quan sát thấy thông tin gì, đề xuất phương án nào, solver thực thi ra sao và nghiệm biến chuyển thế nào. Trọng tâm giải thích đặt vào WHERE và HOW, còn phần WHEN được diễn giải dựa trên trigger và khoảng cách đã ghi nhận.

### 2.1. Khung phân tích và giới hạn suy luận

| Câu hỏi cần làm rõ | Phương pháp xác minh | Giới hạn học thuật |
| :--- | :--- | :--- |
| **Agent đã chọn gì?** | Trích xuất StatePacket đầu vào, phản hồi thô của từng vai trò, action sau khi validate và kết quả thực thi từ operator. Pipeline Multi tách riêng bản ghi của Inspector và Planner. | Không đánh đồng đề xuất ban đầu của LLM với action thực tế sau khi qua bộ lọc fallback hoặc kết quả thực thi của solver. |
| **Lựa chọn có cơ sở hay không?** | Đối chiếu phát biểu của LLM với các bằng chứng thực tế trong region và boundary context tương ứng trong trace. | Việc tìm thấy bằng chứng chỉ chứng minh thông tin có tồn tại trong prompt, không khẳng định LLM thực sự suy luận dựa trên bằng chứng đó. |
| **Tại sao không chọn phương án khác?** | Liệt kê các phương án khả dĩ tại thời điểm đó. Nhóm chỉ đánh giá phương án thay thế khi đã chạy thực nghiệm đối chuẩn trên cùng một trạng thái xuất phát. | Tuyệt đối không suy diễn chủ quan rằng phương án chưa thử là kém hơn hay bất khả thi. |
| **Hành động mang lại tác động gì?** | Phân tách rõ điểm số cá thể nguồn, điểm thử nghiệm tốt nhất (best trial), điểm ứng viên (candidate), trạng thái thu nhận vào quần thể (`selected`) và biến chuyển của Incumbent toàn cục. | Kết quả của một hành động đơn lẻ không đại diện cho xu hướng dài hạn của thuật toán. Điểm $H$ và $S$ được diễn giải theo đúng quy ước phân cấp. |

### 2.2. Kiểm toán Offline từ dữ liệu Trace
- **Nguồn dữ liệu:** Tùy phiên bản và trạng thái run, `controller_trace.jsonl` có thể chứa StatePacket, trigger, nhật ký vai trò, action sau validation, `OperatorOutcome` và incumbent trước/sau. Kiểm tra source của run để xác định payload Planner.
- **Công cụ trích xuất:** Script `python tools/explain_controller_trace.py <đường-dẫn/controller_trace.jsonl>` tự động phân tích và xuất báo cáo Markdown có trích dẫn số dòng JSONL cụ thể.
- **Đặc tính độc lập:** Công cụ phân tích hoàn toàn offline trên file log, không gọi thêm API, không làm thay đổi trạng thái solver, RNG hay ngân sách của các pipeline.

### 2.3. Đánh giá chất lượng lời giải thích

| Tiêu chí | Phương pháp đo đạc |
| :--- | :--- |
| **Tính xác thực (Factual Accuracy)** | Đối chiếu các phát biểu về ràng buộc, lớp học và điểm số với bằng chứng trong trace; báo số claim đúng trên số claim được kiểm tra. Việc kiểm tra claim tự động chưa được triển khai. |
| **Tính hợp lệ của suy luận** | Thống kê số lần LLM bị ảo giác (viện dẫn ràng buộc không tồn tại), nhầm lẫn giữa lựa chọn chủ động với fallback, hoặc tự khẳng định quan hệ nhân quả chưa qua kiểm chứng. |
| **Tính hữu ích thực tiễn** | Đánh giá độ rõ ràng và khả năng hỗ trợ chuyên gia nắm bắt nhanh nguyên nhân chọn vùng và hiệu quả can thiệp so với việc phải đọc file log JSON thô. |
| **Chi phí bổ sung** | Đo đạc chính xác thời gian sinh báo cáo, số token và độ trễ gia tăng nếu yêu cầu LLM sinh thêm phần giải thích trong payload. |

---

## 3. Kế hoạch Thực nghiệm và Bảng Kết quả

Lộ trình thực nghiệm được tổ chức thành 3 giai đoạn rõ ràng:

| Giai đoạn | Mục tiêu chính | Quy mô thực hiện |
| :--- | :--- | :--- |
| **1. Khảo sát Pilot** | Đánh giá sơ bộ 3 pipeline (Baseline, Single, Multi) trên cấu hình ban đầu | 9 instances, 2 seeds |
| **2. Chuẩn hóa StatePacket** | Đo đạc 6 phiên bản, chạy sâu v2, v5, v6 trên 9 instances để chốt cấu hình tối ưu | Bảng dữ liệu theo từng seed |
| **3. Chiến dịch Chính thức** | Đánh giá toàn diện 5 pipeline trên cấu hình chuẩn hóa | 18 instances, 3 seeds (270 runs dự kiến) |

---

### 3.1. Giai đoạn 1: Cấu hình và Kết quả thực nghiệm đã ghi nhận

| Thành phần | Thông số thiết lập |
| :--- | :--- |
| **Seeds & Pipelines** | Seeds 17 và 42, gồm 3 pipeline: Baseline, Single-agent, Multi-agent. |
| **Cấu hình Solver** | Quần thể 20, $H=3$, tối đa 200 thế hệ. Riêng `yach-fal17` và `nbi-spr18` dùng quần thể 12 ($H=2$), còn `bet-spr18` chạy tối đa 230 thế hệ. |
| **Ngân sách đánh giá (Cap)** | 1,200 evaluations thông thường (riêng `mary-fal18` là 1,100 và `bet-spr18` là 2,300). Mọi so sánh chỉ thực hiện giữa các pipeline có cùng evaluation cap. |
| **Controller & Repair** | Bộ repair chung, warmup 5 thế hệ, review interval mặc định 10 (khoảng 3–20), stagnation window 5, diversity 0.2/0.3, tối đa 6 regions (size 5), budget tối đa 24 eval/review, tối đa 12 reviews. |
| **Mô hình LLM** | `gemini-3.5-flash-lite`, timeout 15s, pacing wait 10s. Single gọi 1 lần/review, còn Multi gọi 2 lần/review. |
| **Thời gian chạy** | Chiến dịch gốc có giới hạn thời gian thực (wall-clock cap) theo từng instance. Riêng 2 bộ ba của `tg-spr18` là thử nghiệm bổ sung không khống chế thời gian, dừng khi chạm mốc 1,200 evaluations. |

**Quy ước biểu diễn:** Giá trị trong ô là cặp số `H / S`, trong đó $H$ là số vi phạm ràng buộc cứng (càng thấp càng tốt), $S$ là tổng điểm phạt ràng buộc mềm nội bộ. Chỉ so sánh điểm $S$ khi cả hai nghiệm đều đạt tính khả thi ($H = 0$).

#### Bảng kết quả Giai đoạn 1 (Chi tiết từng Instance và Seed):

| Instance | Seed 17 Baseline | Seed 17 Single | Seed 17 Multi | Seed 42 Baseline | Seed 42 Single | Seed 42 Multi |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `muni-fi-spr16` | 0 / 12,688 | 0 / 12,495 | 0 / 12,462 | 0 / 13,180 | 0 / 13,516 | 0 / 13,699 |
| `muni-fsps-spr17` | 0 / 200,344 | 0 / 182,660 | 0 / 176,712 | 0 / 187,635 | 0 / 190,166 | 0 / 182,205 |
| `yach-fal17` | 2 / 22,016 | 2 / 22,016 | 2 / 21,254 | 3 / 20,563 | 4 / 20,630 | 4 / 20,835 |
| `nbi-spr18` | 0 / 67,454 | 0 / 68,909 | 0 / 68,935 | 0 / 73,078 | 0 / 72,394 | 0 / 72,204 |
| `mary-fal18` | 0 / 22,728 | 0 / 22,623 | 0 / 22,762 | 0 / 23,322 | 0 / 23,169 | 0 / 23,031 |
| `bet-spr18` | 34 / 420,508 | 29 / 420,320 | 28 / 418,470 | 36 / 416,273 | 35 / 417,952 | 33 / 417,285 |
| `pu-cs-fal07` | 0 / 894 | 0 / 744 | 0 / 610 | 0 / 880 | 0 / 820 | 0 / 830 |
| `lums-spr18` | 3 / 745 | 0 / 695 | 3 / 745 | 0 / 914 | 0 / 743 | 0 / 718 |
| `tg-spr18*` | 17 / 74,300 | 16 / 74,070 | 15 / 73,870 | 22 / 77,282 | 21 / 76,682 | 21 / 76,242 |

*\* `tg-spr18` thuộc bộ chạy bổ sung không khống chế thời gian, còn 16 bộ ba còn lại thuộc chiến dịch gốc. Tổng cộng 54 runs trên hai điều kiện thời gian được ghi nhận minh bạch.*

#### Thống kê đối đầu trực tiếp:

| Cặp so sánh | 16 bộ ba gốc (Thắng / Thua / Hòa / Chưa phân định) | 2 bộ ba `tg` (Thắng / Thua / Hòa / Chưa phân định) | Tổng 18 bộ ba (Thắng / Thua / Hòa / Chưa phân định) | Số instance thắng $\ge 1$ seed | Số instance thắng cả 2 seeds |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single vs Baseline** | 11 / 4 / 0 / 1 | 2 / 0 / 0 / 0 | **13 / 4 / 0 / 1** | 8 / 9 | 5 / 9 |
| **Multi vs Baseline** | 10 / 4 / 0 / 2 | 2 / 0 / 0 / 0 | **12 / 4 / 0 / 2** | 8 / 9 | 4 / 9 |
| **Multi vs Single** | 9 / 5 / 0 / 2 | 1 / 0 / 0 / 1 | **10 / 5 / 0 / 3** | 8 / 9 | 2 / 9 |

- **Quy tắc phân định:** "Chưa phân định" nghĩa là cả hai nghiệm đều có $H > 0$ bằng nhau (không dùng $S$ để tuyên bố thắng thua). "Hòa" là khi cùng đạt $H = 0$ và có điểm $S$ bằng nhau.
- Số trường hợp đạt tính khả thi hoàn toàn ($H = 0$) trên 18 lần chạy: Baseline đạt 11, Single đạt 12, Multi đạt 11.

#### Thống kê tiêu tốn tài nguyên (Runtime giây / API calls / Native tokens):

| Nhóm bài toán | Baseline | Single-agent | Multi-agent |
| :--- | :---: | :---: | :---: |
| **16 bộ ba gốc** | 4,673 s / 0 calls / 0 tokens | 4,600 s / 111 calls / 778,308 tokens | 5,823 s / 250 calls / 1,772,823 tokens |
| **2 bộ ba `tg` bổ sung** | 5,818 s / 0 calls / 0 tokens | 6,946 s / 10 calls / 68,815 tokens | 7,627 s / 18 calls / 126,092 tokens |

---

### 3.2. Giai đoạn 2: Khảo sát và Chuẩn hóa StatePacket

- Vòng đầu đã hoàn thành khảo sát 6 phiên bản cấu hình (`v1` $\rightarrow$ `v6`). Tiếp tục hoàn thiện bảng số liệu cho **3 phiên bản tiêu biểu: v2, v5 và v6 trên toàn bộ 9 instances**.
- Dữ liệu được ghi nhận độc lập theo từng seed, không gộp chung seed hoặc thay thế dữ liệu khuyết từ cấu hình khác. Mỗi ô thể hiện kết quả `B=(H,S), Single=(H,S), Multi=(H,S)`.
- Mục tiêu: Chốt một cấu hình StatePacket chính thức trước khi bước vào Giai đoạn 3 dựa trên các tiêu chí: tính hợp lệ cấu trúc, tỷ lệ giảm vi phạm $H$, điểm $S$, độ ổn định lỗi API (tránh `MAX_TOKENS`), runtime và số token tiêu hao.

| Instance | v2: B / Single / Multi | v5: B / Single / Multi | v6: B / Single / Multi |
| :--- | :---: | :---: | :---: |
| `muni-fi-spr16` | — | — | — |
| `muni-fsps-spr17` | — | — | — |
| `yach-fal17` | — | — | — |
| `nbi-spr18` | — | — | — |
| `mary-fal18` | — | — | — |
| `bet-spr18` | — | — | — |
| `pu-cs-fal07` | — | — | — |
| `lums-spr18` | — | — | — |
| `tg-spr18` | — | — | — |

---

### 3.3. Giai đoạn 3: Chiến dịch Thực nghiệm Chính thức (5 Pipelines, 18 Instances)

- **Quy mô bài toán:** Gồm 9 instances cũ (tập đã dùng khảo sát) và bổ sung thêm **9 instances mới** từ bộ ITC 2019 được chọn trước theo các tiêu chí: quy mô số lớp, số phòng, độ phức tạp của ràng buộc phân phối và phân lớp sinh viên.
- **Tính đóng băng tham số:** Chốt cố định StatePacket, tập toán tử, prompt template, cơ chế trigger WHEN, thông số solver/repair và quy tắc retry trước khi bấm máy. Cả 5 pipeline chạy trên cùng một seed và cùng ngân sách evaluation trong từng bài toán.
- **Quy mô thực thi:** 18 instances $\times$ 5 pipelines = **90 runs / seed**, chạy trên **3 seeds** (17, 42 và seed thứ ba) $\rightarrow$ Tổng cộng **270 runs chính thức** (chưa bao gồm các lần retry lỗi mạng).

#### Bảng tổng hợp dữ liệu Giai đoạn 3 (Mỗi ô điền kết quả của 3 seeds: `S1: H/S, S2: H/S, S3: H/S`):

| Instance | Nhóm bài | Baseline | Random | Greedy | Single-agent | Multi-agent |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `muni-fi-spr16` | Hiện tại | — | — | — | — | — |
| `muni-fsps-spr17` | Hiện tại | — | — | — | — | — |
| `yach-fal17` | Hiện tại | — | — | — | — | — |
| `nbi-spr18` | Hiện tại | — | — | — | — | — |
| `mary-fal18` | Hiện tại | — | — | — | — | — |
| `bet-spr18` | Hiện tại | — | — | — | — | — |
| `pu-cs-fal07` | Hiện tại | — | — | — | — | — |
| `lums-spr18` | Hiện tại | — | — | — | — | — |
| `tg-spr18` | Hiện tại | — | — | — | — | — |
| *Mới 01* | Mở rộng | — | — | — | — | — |
| *Mới 02* | Mở rộng | — | — | — | — | — |
| *Mới 03* | Mở rộng | — | — | — | — | — |
| *Mới 04* | Mở rộng | — | — | — | — | — |
| *Mới 05* | Mở rộng | — | — | — | — | — |
| *Mới 06* | Mở rộng | — | — | — | — | — |
| *Mới 07* | Mở rộng | — | — | — | — | — |
| *Mới 08* | Mở rộng | — | — | — | — | — |
| *Mới 09* | Mở rộng | — | — | — | — | — |

#### Mẫu nhật ký chi tiết cho từng Run thực nghiệm:

| Instance | Seed | Pipeline | Trạng thái | H / S nội bộ | H / S official | Evaluations / Cap | Thời gian (s) | API Calls | Native Tokens | Attempt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| — | — | — | — | — | — | — | — | — | — | — |

---

## 4. Hệ thống Biểu đồ Trực quan hóa

Hệ thống biểu đồ phục vụ báo cáo và bài báo khoa học được phân tách thành 3 nhóm trọng tâm:

| Nhóm hình | Cấu trúc hiển thị & Nguồn dữ liệu | Câu hỏi khoa học cần trả lời |
| :--- | :--- | :--- |
| **1. Chất lượng nghiệm cuối (Final Quality)** | **Biểu đồ thanh ngang (Horizontal Bar Chart):** Thể hiện kết quả trên từng instance/seed cho 5 pipeline. Trục chính thể hiện vi phạm cứng $H$, chú thích điểm mềm $S$. Khi toàn bộ các pipeline đều đạt $H = 0$, biểu đồ chuyển sang so sánh trực tiếp điểm $S$. Nguồn: `summary.json`. | Phương pháp nào đạt tỷ lệ khả thi cao nhất? Khi đã khả thi, phương pháp nào tối ưu hóa điểm phạt mềm tốt hơn? |
| **2. Tốc độ hội tụ và Điểm can thiệp (Convergence & Reviews)** | **Đồ thị đường bậc thang (Step Curves):** Biểu diễn sự biến thiên của Incumbent $H$ theo **Evaluations**, **Thời gian thực (giây)** và **Thế hệ**. Đánh dấu trực quan các thời điểm Review Trigger, loại toán tử được chọn, và kết quả thực thi (`selected` / `no improvement`) trích xuất từ `controller_trace.jsonl`. Nguồn: `anytime_trace.jsonl`. | Tốc độ cải thiện nghiệm diễn ra như thế nào? Tại các thời điểm LLM can thiệp, đường cong có tạo "bước nhảy vọt" (sharp drop) rõ rệt so với Baseline và Random hay không? |
| **3. Chi phí thời gian và Tài nguyên API** | **Biểu đồ cột nhóm (Grouped Bar Chart):** Tổng hợp runtime thực tế, số lượt gọi API và tổng lượng native tokens tiêu tốn, kèm đồ thị phân bố token/độ trễ trên từng lượt review để phân tích biến động chi phí. | Mức độ cải thiện chất lượng nghiệm có tương xứng với chi phí token và độ trễ chờ API giữa Single-agent và Multi-agent hay không? |

### Nguyên tắc trình bày biểu đồ chuẩn học thuật:
- **Tính nhất quán màu sắc:** Sử dụng bảng màu cố định đại diện cho từng pipeline xuyên suốt toàn bộ các hình vẽ.
- **Minh bạch số lượng mẫu:** Với 3 seeds thực nghiệm, ghi chú rõ $n=3$ và hiển thị từng đường chạy riêng biệt thay vì tính trung bình gộp để tránh gây hiểu nhầm về mặt thống kê.
- **Tách biệt $H$ và $S$:** Không gộp vi phạm cứng và điểm phạt mềm thành một giá trị vô hướng duy nhất. Điểm $S$ có thể tạm thời tăng lên trong quá trình giải thuật tìm cách triệt tiêu vi phạm $H$.
- **Đánh dấu điểm can thiệp trung thực:** Vị trí marker thể hiện chính xác thời điểm ghi nhận review. Kết quả cải thiện nghiệm phải được đối chiếu khách quan với Random và Greedy trước khi đưa ra nhận định về năng lực của LLM.
- **Chuẩn hóa đo lường tài nguyên:** Token tích lũy chỉ tăng hoặc đi ngang. Biểu đồ chi phí review phải dùng chỉ số token trên từng lượt gọi (`tokens_per_review`). Độ trễ đo đạc đã bao gồm thời gian chờ pacing theo quy định của nhà cung cấp API.
