# Cơ sở cho bản thiết kế thực nghiệm v2

## Scope / terminology

- Phạm vi: thiết kế lại hai thí nghiệm **NSGA-III + single-agent** và **NSGA-III + Inspector–Planner**, với NSGA-III không controller là mốc chung; không sửa trực tiếp proposal gốc.
- Tên cố định: `StatePacket` (summary do code tính), `RegionCandidate` (vùng hợp lệ được code tạo), `ActionPlan` (quyết định có cấu trúc), `Inspector` (WHERE), `Planner` (HOW+WHEN), `review interval` (số thế hệ tối đa tới lần review kế tiếp), `routine repair` (bước chung của cả ba phương pháp), `improvement operator` (can thiệp bổ sung của hai controller).
- `AUTHOR_INPUT_NEEDED` chỉ các lựa chọn nhóm chưa chốt; tham số ký hiệu không phải kết quả hay cấu hình tối ưu.

## Research canon

1. Proposal hiện tại định nghĩa ba phương pháp, dùng chung NSGA-III/representation/evaluator/feasibility checker/routine repair, cùng ITC 2019 instances/seeds/wall-clock; hai controller dùng chung improvement operators. [Proposal: PDF pp. 6–8]
2. Single-agent quyết định WHERE/HOW/WHEN trong một lần gọi; Inspector–Planner có đúng hai agent, Inspector cho WHERE, Planner cho HOW+WHEN. [Proposal: PDF pp. 7–8]
3. Solver kiểm chứng quyết định, áp dụng toán tử, loại nghiệm vi phạm hard constraints và đưa ứng viên hợp lệ qua selection của NSGA-III. [Proposal: PDF pp. 7–8]
4. Đầu vào summary, định nghĩa vùng, mục tiêu NSGA-III, operator catalogue, ngưỡng trigger, schema và chính sách lỗi **chưa được đặc tả**. Không được viết như đã triển khai.
5. LAOS cập nhật operator trên lịch cố định và dùng state/credit/replay; không cung cấp trực tiếp WHERE hoặc adaptive review timing cho UCTP. [P005: PDF pp. 3–8]
6. DRAGON dùng active/static decomposition, constraint-aware compression và reconstruction trên một nghiệm; không thử Inspector–Planner trên quần thể NSGA-III. [P012: PDF pp. 3–8]
7. MOTIF tối ưu mã strategy **offline** bằng hai player/MCTS; không đo lợi ích hai vai Inspector–Planner online. [P047: PDF pp. 3–7]
8. LLM-LNS tiến hóa heuristic offline; adaptive rule lúc solve đổi neighborhood size khi stagnation/timeout, không gọi LLM để chọn review interval. Appendix nêu chi phí evolution nằm ở training stage, không ảnh hưởng inference-time. [P002: PDF pp. 4–6, 47–48]

## Evidence table

| Claim usable in design | Source | Strength / risk |
|---|---|---|
| Compact state + operator outcomes có thể đưa vào prompt để chọn HOW | P005, PDF pp. 4–5, Fig. 2/Table 1 | Evidence-backed trong EA đơn mục tiêu; chuyển sang NSGA-III là thiết kế mới. |
| Cần boundary constraints khi sửa cục bộ | P012, PDF pp. 3–5, Fig. 2 | Evidence-backed trong DRAGON; biểu diễn UCTP chưa có. |
| Tách vai có thể hữu ích nhưng cần kiểm định riêng | P047, PDF pp. 3–7 | Hypothesis cho Inspector–Planner; paper không kiểm định hai vai này. |
| Stagnation có thể dùng làm event do code đo | P002, PDF p. 6 | Evidence-backed như ALNS rule; không chứng minh adaptive LLM timing. |
| Hai controller phải hạch toán cả overhead LLM | Proposal, PDF p. 8; P005, PDF p. 8; P012, PDF p. 5 | Design requirement / cost precedent. |
| Multi-agent tốt hơn single-agent trên ITC 2019 | Chưa có | Hypothesis; không được nêu như kết quả. |

## Argument map

Question 1: Với cùng NSGA-III, dữ liệu và wall-clock, thêm controller WHERE/HOW/WHEN có cải thiện chất lượng–thời gian so với không controller không? E1 đối chiếu baseline với single-agent; kết quả chỉ đánh giá **gói can thiệp**, chưa cô lập vai trò riêng của LLM.

Question 2: Với cùng solver, trigger, `StatePacket`, candidate regions, operator library, schema hành động và ngân sách, tách WHERE sang Inspector rồi HOW+WHEN sang Planner có lợi hơn một lời gọi gộp không? E2 đối chiếu single-agent với Inspector–Planner; overhead của lời gọi thứ hai là một phần thực của phương pháp và phải tính.

Counterargument chính: thắng baseline có thể do thêm improvement operators; thắng single-agent có thể do hai lần suy luận/nhiều token chứ không phải phân vai. Vì vậy phải báo cáo cả chất lượng ở wall-clock cố định lẫn số lời gọi/token/can thiệp và giới hạn diễn giải.

## Section contract for design v2

- Allowed: workflow khả thi đề xuất, giả thuyết, tham số cần pilot/chốt, phép đo và điểm mơ hồ.
- Forbidden: tuyên bố đây là cơ chế đã triển khai, tự đặt thông số ITC 2019 chưa đọc, khẳng định SOTA/novelty hoặc lợi ích multi-agent chưa đo.
- Validation: mọi gợi ý lấy từ paper phải có locator; mỗi quyết định thiết kế mới phải ghi rõ là khuyến nghị chứ không phải kết quả paper.
