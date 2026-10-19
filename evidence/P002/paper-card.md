# LLM-LNS — Paper Card

> Source coverage: Full local paper PDF (50 pages), main text and relevant appendix passages.
> Extraction confidence: Mixed (bundled figure/table detector missed main-text objects; they were inventoried manually from page text).
> Locator mode: page-grounded.
> Primary analytical lens: methods.
> Secondary analytical lens: None.
> Context verification: Paper-only.
> Card completeness: Complete for the mechanism and main results, partial for appendix details.

## 01 Thông tin cơ bản

[Paper] Huigen Ye, Hua Xu, An Yan và Yaoyang Cheng, “Large Language Model-driven Large Neighborhood Search for Large-Scale MILP Problems,” ICML/PMLR 267 (2025). Liên quan tới WHERE và phản ứng với stagnation, nhưng không điều khiển LLM online trong EA. [Paper: PDF p. 1]

## 02 Tóm tắt một câu

[Paper] Paper tiến hóa *offline* một heuristic chấm điểm biến bằng LLM trên bài MILP nhỏ, rồi dùng heuristic đó trong LNS chạy trên bài lớn với quy tắc tăng/giảm kích thước vùng theo tiến độ và thời gian. [Paper: PDF pp. 4–6, Figure 1/Section 3.2]

## 03 Câu hỏi nghiên cứu

[Paper] Có thể tạo chiến lược chọn vùng LNS chuyển từ MILP nhỏ sang lớn mà không cần thiết kế tay hoặc huấn luyện nhiều dữ liệu không? [Paper: PDF pp. 1–3]

## 04 Bối cảnh và đường phát triển

[Paper-framed; external verification not performed] Paper đặt Random-LNS, ACP và ML-LNS làm các đối chứng; LLM được dùng để tiến hóa chiến lược chọn vùng. Không xem các nhận định novelty trong Introduction là đã được xác minh độc lập. [Paper: PDF pp. 2–3]

## 05 Nút thắt mà paper nêu

[Paper] LNS phụ thuộc chiến lược chọn neighborhood; thủ công đòi hỏi chuyên môn, RL/IL tốn huấn luyện/dữ liệu. [Paper: PDF pp. 1–3]

## 06 Ý tưởng cốt lõi

[Paper] Outer LLM layer tiến hóa prompt, inner layer tiến hóa heuristic code/thought; heuristic xếp hạng biến, solver truyền thống giải bài toán con. [Analysis] “WHEN” trong paper là quy tắc ALNS dựa trên stagnation/time để đổi **kích thước neighborhood**, không phải LLM quyết định khoảng gọi lại. [Paper: PDF pp. 4–6]

## 07 Tổng quan phương pháp

[Paper] Pha huấn luyện: tiến hóa strategy trên instances nhỏ. Pha chạy: heuristic chấm điểm biến, chọn top-k, giải subproblem, cập nhật nghiệm; nếu cải thiện dưới `ε` trong `p` lượt, mở rộng `k`; nếu solve subproblem vượt thời gian, thu nhỏ `k`. [Paper: PDF pp. 4–6, Figure 1/Section 3.2]

## 08 Phân rã module

- [Paper] Inner layer sinh/evaluate heuristic strategy. [Paper: PDF pp. 4–5]
- [Paper] Outer layer sinh prompt strategy; differential memory dùng fitness của các parent để định hướng sinh code. [Paper: PDF pp. 4–6]
- [Paper] ALNS dùng variable scoring và deterministic neighborhood-size adaptation; subsolver giải vùng đã chọn. [Paper: PDF p. 6]

## 09 Công thức và ký hiệu thiết yếu

[Paper] Equation 1 là MILP; Equation 2 và Equation 3 mô tả strategy/memory và generation. Equation 4 định nghĩa `N(t)={x_i | rank(s_i(t))≤k}`. Appendix Equation 11, Equation 12 và Equation 13 lặp lại chọn vùng và luật giảm/tăng `k`; Equation 5, Equation 6, Equation 7, Equation 8, Equation 9, Equation 10 là công thức các ví dụ heuristic; Equation 14, Equation 15, Equation 16, Equation 17 định nghĩa bài thử MILP. `Equation 20` tại PDF p. 42 là false positive do parser đọc nhãn bảng, không phải phương trình. Đây là điều chỉnh neighborhood, không phải gọi LLM mới trong thời gian chạy. [Paper: PDF pp. 2, 5–6, 15–23, 42]

## 10 Thiết kế thực nghiệm và chuỗi bằng chứng

[Paper] Figure 1 trình bày kiến trúc, Table 1 đối chiếu tính năng heuristic evolution, Tables 2–3 là kết quả bin packing/TSP. Table 4 so mục tiêu MILP; Figure 3 và Table 5 mô tả convergence/primal integral. Các đối chứng gồm Random-LNS, ACP, CL-LNS, Gurobi/SCIP và ML frameworks; Gurobi làm subsolver cho các nhánh LNS. Appendix G ablate dual layer/differential (Tables 13–15); Appendix I nêu giới hạn. Bộ prepare không phát hiện figure/table vì layout PDF, nên các mục chính được đọc trực tiếp từ PDF pp. 3–9 và appendix pp. 41–49. [Paper: PDF pp. 3–9, 41–49]

## 11 Diễn giải kết luận đúng mức

[Paper] Paper báo cáo chất lượng tốt trên các MILP được thử và chi phí tiến hóa chiến lược nằm ở pha huấn luyện. [Analysis] Không có bằng chứng rằng gọi LLM online theo event tốt hơn fixed review trong UCTP. Một số câu diễn giải kết quả trong appendix dùng “lower is better” cho cả bài maximization; cần dựa vào định nghĩa mục tiêu từng bài chứ không sao chép nhận xét tổng quát. [Paper: PDF pp. 7–9, 40–41]

## 12 Giới hạn tác giả thừa nhận

[Paper] Framework hiện tập trung MILP và chưa tích hợp đầy đủ domain knowledge/optimization expertise; tác giả nêu khả năng mở rộng sang bài toán khác là hướng tương lai. [Paper: PDF p. 49, Appendix I]

## 13 Phân tích phê bình

[Analysis] Việc tách chi phí huấn luyện khỏi inference khác căn bản với controller LLM được gọi trong wall-clock run. Nếu lấy paper này làm prior art cho adaptive WHEN, cần ghi rõ nó chỉ có deterministic stagnation rule cho neighborhood size. [Paper: PDF pp. 4–6, 47–48]

## 14 Kiến thức có thể chuyển giao

[Analysis] Dùng score để giới hạn candidate region, ràng buộc kích thước neighborhood, giữ subsolver xác thực. Có thể mượn quy tắc stagnation làm một control không-LLM, nhưng không phải thay thế trực tiếp lịch review của LLM. [Paper: PDF p. 6]

## 15 Liên hệ với đề tài

[Analysis] Đề tài dùng NSGA-III online với LLM chọn region/operator/review interval; LLM-LNS là MILP LNS với code heuristic đã tiến hóa offline và thời điểm điều chỉnh neighborhood theo luật. [Paper: PDF pp. 4–6]

## 16 Ý tưởng nghiên cứu

[Hypothesis] LLM chọn review interval có thể tốt hơn một lịch/event rule thuần túy chỉ khi lợi ích thích nghi vượt latency. Delta: thay fixed/deterministic interval bằng action WHEN có kiểm chứng và giới hạn; giữ cùng WHERE/HOW. Validation: event-rule controller không-LLM hoặc interval cố định với cùng operator/target policy và wall-clock. Falsifier: không cải thiện chất lượng theo thời gian. Failure modes: trigger nhiễu, chi phí gọi LLM, interval tự gia hạn quá lâu. Innovation status: unverified.
