# DRAGON — Paper Card

> Source coverage: Full local paper PDF (21 pages); supplementary materials in the same PDF not exhaustively audited.
> Extraction confidence: High for main text.
> Locator mode: page-grounded.
> Primary analytical lens: methods.
> Secondary analytical lens: None.
> Context verification: Paper-only.
> Card completeness: Complete for the main paper, partial for supplements.

## 01 Thông tin cơ bản

[Paper] Shengkai Chen và cộng sự, “DRAGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization,” AAMAS 2026, DOI 10.65109/SBAY6258. Liên quan tới WHERE và ràng buộc biên, nhưng không phải NSGA-III hay UCTP. [Paper: PDF p. 1]

## 02 Tóm tắt một câu

[Paper] DRAGON chọn một vùng hoạt động của một nghiệm lớn, nén thành bài toán con, tái dựng và đưa kết quả qua kiểm tra khả thi trước khi tích hợp. [Paper: PDF pp. 3–5, Algorithm 1/Figure 2]

## 03 Câu hỏi nghiên cứu

[Paper] LLM có tìm được vùng tiềm năng trong một nghiệm lớn và tái giải bài toán con với ràng buộc bổ sung không? [Paper: PDF p. 2]

## 04 Bối cảnh và đường phát triển

[Paper-framed; external verification not performed] Paper đối chiếu direct prompting và heuristic thủ công, rồi đề xuất phân rã/tái dựng để vượt giới hạn ngữ cảnh. Không suy ra một kiểm chứng lịch sử độc lập. [Paper: PDF pp. 1–3]

## 05 Nút thắt mà paper nêu

[Paper] Prompt toàn bài toán lớn vượt khả năng LLM; vùng con có thể phá ràng buộc toàn cục nếu thiếu thông tin biên. [Paper: PDF pp. 2–4]

## 06 Ý tưởng cốt lõi

[Paper] Decomposer chọn active/static segments; `Compress` giữ metadata liên quan và mã hóa ràng buộc của vùng tĩnh; Reconstructor tạo nghiệm con; checker và bước tích hợp bảo vệ tính khả thi. [Analysis] Đề tài UCTP nên mượn hợp đồng vùng/biên, không đồng nhất Inspector với toàn bộ DRAGON. [Paper: PDF pp. 3–5, Figure 2]

## 07 Tổng quan phương pháp

[Paper] Đầu vào metadata và một nghiệm hiện tại. Mỗi vòng: `D` → `Compress` → `R` → `Accept` → `Integration`; giới hạn thời gian và số lần bị từ chối. DRAGON dùng hai tác tử cho decomposition và reconstruction; paper cũng thử solver thay reconstruction trong ablation. [Paper: PDF pp. 3–4, Algorithm 1; PDF pp. 6–7, Figure 6]

## 08 Phân rã module

- [Paper] Decomposer trả active/static segment; hình minh họa TSP ở Figure 1. [Paper: PDF pp. 1, 3]
- [Paper] Compression sinh metadata và constraints cục bộ; Figure 2 cho thấy state passing. [Paper: PDF p. 4]
- [Paper] Reconstructor dùng constraints và memory phản hồi infeasible; Figure 3 mô tả vòng sửa. [Paper: PDF pp. 4–5]
- [Paper] Checker kiểm tra ràng buộc, `Accept` cho phép nghiệm xấu hơn theo xác suất; không nên copy nguyên quy tắc này vào NSGA-III. [Paper: PDF pp. 3, 5]

## 09 Công thức và ký hiệu thiết yếu

[Paper] Equation 1 mô tả cập nhật nghiệm; Equation 2 là `Accept` kiểu simulated annealing. Equation 3 định nghĩa `D(M,S)=(a,s)`; Equation 4 định nghĩa `Compress(M,a,s)=(M',S',C)`; Equation 5 định nghĩa `R(M',S',C)=S'new`. [Paper: PDF pp. 3–4, Equations 1–5]

## 10 Thiết kế thực nghiệm và chuỗi bằng chứng

[Paper] TSP/TSPLIB, CVRP/CVRPLIB, BPP/Weibull-5k, MKP tổng hợp; metrics gồm gap, runtime, token và API calls; giới hạn 1 giờ và dừng sau 5 vòng không cải thiện. Table 1, Figure 4 và Figure 5 là routing gap; Table 2/Figure 6 là ma trận 3×3 decomposition–reconstruction; Figure 7 là cost/model, Table 3 và Figure 8 là BPP/MKP. Figure 1, Figure 2 và Figure 3 minh họa kiến trúc, compression và feasibility memory. [Paper: PDF pp. 1, 4–8]

## 11 Diễn giải kết luận đúng mức

[Paper] DRAGON đạt nghiệm khả thi trên các benchmark được thử; Figure 6 cho thấy hiệu quả phụ thuộc tổ hợp decomposer/reconstructor, và paper nêu solver reconstruction có thể tốt nhưng đòi hỏi mô hình hóa constraints. [Analysis] Không có thí nghiệm phân vai Inspector–Planner hay điều khiển population-based online. [Paper: PDF pp. 6–8]

## 12 Giới hạn tác giả thừa nhận

[Paper] Chất lượng nhạy với phân đoạn và prompt; lặp qua các vùng tốn thời gian/ngữ cảnh; dùng solver ngoài cần chuyên môn mô hình hóa. [Paper: PDF p. 8, Section 4.5]

## 13 Phân tích phê bình

[Analysis] Tính hiệu quả của decomposition trong một nghiệm không đảm bảo cách chọn một vùng từ cả quần thể NSGA-III. Đặc biệt cần kiểm tra ràng buộc nối vùng và cách phân phối lợi ích giữa các mục tiêu. [Paper: PDF pp. 3–4, Figure 2]

## 14 Kiến thức có thể chuyển giao

[Analysis] Định nghĩa target có ID ổn định, giữ vùng ngoài tĩnh khi thực thi, đưa constraint boundary vào payload và trả lỗi checker có cấu trúc. [Paper: PDF pp. 3–5]

## 15 Liên hệ với đề tài

[Analysis] Inspector đề xuất vùng và boundary tương tự nhánh decomposition, nhưng Planner của proposal chọn một operator có sẵn và review interval; nó không trực tiếp sinh nghiệm con như Reconstructor gốc. [Paper: PDF pp. 3–4]

## 16 Ý tưởng nghiên cứu

[Hypothesis] Chẩn đoán vùng có ràng buộc biên có thể giảm can thiệp vô ích so với chọn vùng ngẫu nhiên trong UCTP. Delta: áp dụng cách đóng gói active/static vào lịch học đa mục tiêu với solver operator. Validation: cố định operator/timing và thay WHERE bằng chọn ngẫu nhiên; falsifier: không cải thiện dưới cùng ngân sách. Failure modes: cluster sai, boundary quá lớn, can thiệp không còn rẻ. Innovation status: unverified.
