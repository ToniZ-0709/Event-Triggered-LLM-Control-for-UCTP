# MOTIF — Paper Card

> Source coverage: Full local paper PDF (23 pages), main text and selected appendix inventory.
> Extraction confidence: High.
> Locator mode: page-grounded.
> Primary analytical lens: methods.
> Secondary analytical lens: None.
> Context verification: Paper-only.
> Card completeness: Complete for the main argument, partial for appendix details.

## 01 Thông tin cơ bản

[Paper] Nguyen Viet Tuan Kiet và cộng sự, “MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework,” arXiv:2508.03929v2 (2025 preprint; PDF notes AAAI 2026 copyright). Liên quan tới đánh giá cấu trúc multi-agent, nhưng là thiết kế solver offline. [Paper: PDF p. 1]

## 02 Tóm tắt một câu

[Paper] MOTIF dùng hai LLM luân phiên trong MCTS để sửa mã nhiều thành phần solver, đánh giá từng ứng viên và cập nhật baseline trong quá trình thiết kế thuật toán offline. [Paper: PDF pp. 3–5, Figure 2]

## 03 Câu hỏi nghiên cứu

[Paper] Có thể tự động tối ưu nhiều strategy phụ thuộc lẫn nhau hiệu quả hơn việc chỉ thiết kế một heuristic không? [Paper: PDF pp. 1–3]

## 04 Bối cảnh và đường phát triển

[Paper-framed; external verification not performed] Paper đặt AHD và LLM hyper-heuristics làm nền, rồi mở sang multi-strategy design và turn-based self-play. Đây là lịch sử do paper trình bày, không phải xác minh độc lập. [Paper: PDF pp. 1–2]

## 05 Nút thắt mà paper nêu

[Paper] Chỉ sửa một thành phần bỏ qua phụ thuộc giữa các thành phần; prompt một chiều ít phản hồi tương tác. [Paper: PDF pp. 1–2, Figure 1]

## 06 Ý tưởng cốt lõi

[Paper] Hai pha: cạnh tranh theo từng strategy với baseline động, sau đó refinement toàn hệ với baseline cố định; hai player luân phiên tạo mã theo Counter/Learning/Innovation. [Analysis] Đây không phải bằng chứng trực tiếp rằng Inspector và Planner chuyên biệt tốt hơn một agent online. [Paper: PDF pp. 3–5]

## 07 Tổng quan phương pháp

[Paper] Đầu vào: solver và các strategy có interface chung; đầu ra: mã implementation đã được chọn sau nhiều lượt đánh giá trên tập huấn luyện. Outer UCB chọn strategy tree; inner CMCTS chọn operator prompt; LLM sinh code; evaluator chạy solver; Q-value cập nhật cây. [Paper: PDF pp. 3–5, Equations 1–8]

## 08 Phân rã module

- [Paper] Strategy-space/outer controller quyết định thành phần nào sửa. [Paper: PDF p. 3]
- [Paper] Two-player CMCTS giữ implementation, lịch sử và relative performance của từng player; Figure 2 mô tả luồng. [Paper: PDF p. 4]
- [Paper] Counter, Learning, Innovation là *prompt operators để sinh code*, không phải repair operators trên thời khóa biểu. [Paper: PDF p. 4]
- [Paper] Phase 2 cung cấp full-system visibility để kiểm tra synergy. [Paper: PDF p. 5]

## 09 Công thức và ký hiệu thiết yếu

[Paper] `Π=(π1,...,πK)` là bộ strategy; mục tiêu tối thiểu kỳ vọng cost của solver với `Π`; outer/inner UCB phân bổ lượt thử; Q-value kết hợp cải thiện tuyệt đối và tương đối. Equation 1, Equation 2, Equation 3 là định nghĩa solver/objective; Equation 4, Equation 5 là UCB; Equation 6, Equation 7, Equation 8 là prompt/reward/refinement. [Paper: PDF pp. 3–5]

[Paper] Inventory appendix (không thiết yếu cho controller UCTP): Equation 9, Equation 10, Equation 11, Equation 12, Equation 13, Equation 14, Equation 15, Equation 16, Equation 17, Equation 18, Equation 19, Equation 20 mô tả các domain objective/constraints; Equation 21, Equation 22, Equation 23, Equation 24, Equation 25 là phân tích reward; Equation 26, Equation 27, Equation 28, Equation 29, Equation 30, Equation 31, Equation 32, Equation 33, Equation 34 là định nghĩa đo diversity trong appendix. Không có công thức event-triggered review cho NSGA-III online. [Paper: PDF pp. 11–12, 17, 20–21]

## 10 Thiết kế thực nghiệm và chuỗi bằng chứng

[Paper] Figure 3 so sánh chất lượng ACO và số lần đánh giá; Table 1 là single-strategy GLS; Table 2 là single/multi-strategy trên TSP/CVRP/BPP; Figure 4 là convergence; Table 3 mô tả prompt-operator diversity; Table 4 là ablation. Figure 1 và Figure 2 giải thích kiến trúc. Appendix Table 5, Table 6, Table 7, Table 8 ghi các strategy được tối ưu, benchmark/hyperparameter. Các bảng chính chủ yếu báo cáo 3 runs; paper dùng dữ liệu train nhỏ và test held-out. [Paper: PDF pp. 1, 4–7, 14, 19–20]

## 11 Diễn giải kết luận đúng mức

[Paper] Trong các cấu hình được thử, thiết kế nhiều strategy với two-player search có kết quả tốt hơn một số đối chứng AHD. [Analysis] Không thể lấy kết quả này làm dự báo cho online Inspector–Planner trong UCTP; đối tượng tối ưu, quyền của agent và thời điểm gọi đều khác. [Paper: PDF pp. 5–7]

## 12 Giới hạn tác giả thừa nhận

No explicit author-acknowledged limitation was found in the main text inspected. [Paper: PDF pp. 1–7]

## 13 Phân tích phê bình

[Analysis] Multi-agent so với single-agent trong MOTIF bị gắn với MCTS, multi-strategy và baseline update; không cô lập riêng hiệu ứng “hai LLM”. Do đó proposal cần giữ interface/action/triggers cố định nếu muốn kiểm định lợi ích tách Inspector–Planner. [Paper: PDF pp. 3–7]

## 14 Kiến thức có thể chuyển giao

[Analysis] Cần tách vai trò và thông tin mỗi vai nhận; ghi log chi phí và outcome của từng lời gọi; thí nghiệm phải cô lập cơ chế cụ thể thay vì viện dẫn lợi ích multi-agent chung. [Paper: PDF pp. 4–7]

## 15 Liên hệ với đề tài

[Analysis] MOTIF chỉ hỗ trợ tính hợp lý của việc *đặt câu hỏi* về tách vai trò. Đề tài khác căn bản: kiểm soát NSGA-III online, không sinh mã strategy. [Paper: PDF pp. 3–5]

## 16 Ý tưởng nghiên cứu

[Hypothesis] Inspector–Planner có thể cho quyết định vùng/operator nhất quán hơn một LLM làm cả ba, nhưng overhead hai lời gọi có thể làm kết quả cuối kém hơn. Delta: tách diagnostic và planning trong controller online, không dùng MCTS/code generation. Validation: cùng model, input/action space, triggers, operators và wall-clock; báo cáo cả chất lượng và latency. Falsifier: không có lợi ích sau hạch toán chi phí. Failure modes: hai lời gọi chậm, handoff mất thông tin, bất đồng giữa hai vai. Innovation status: unverified.
