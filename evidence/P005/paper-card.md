# LAOS — Paper Card

> Source coverage: Full local paper PDF (10 pages); supplement not supplied.
> Extraction confidence: High.
> Locator mode: page-grounded.
> Primary analytical lens: methods.
> Secondary analytical lens: None.
> Context verification: Paper-only.
> Card completeness: Complete relative to supplied source.

## 01 Thông tin cơ bản

[Paper] Yisong Zhang và Guoxing Yi, “LAOS: Large Language Model-Driven Adaptive Operator Selection for Evolutionary Algorithms,” GECCO 2025, DOI 10.1145/3712256.3726450. Vị trí đối với đề tài: điểm tham chiếu trực tiếp cho quyết định HOW trong EA; không phải bản triển khai UCTP/NSGA-III. [Paper: PDF p. 1]

## 02 Tóm tắt một câu

[Paper] LAOS dùng trạng thái quần thể, credit của toán tử và kinh nghiệm offline/online để LLM chọn toán tử cho giai đoạn tối ưu tiếp theo theo lịch cập nhật cố định. [Paper: PDF p. 3, Figure 1; PDF p. 4, Algorithm 1]

## 03 Câu hỏi nghiên cứu

[Paper] Có thể chọn toán tử tiến hóa trực tuyến mà không cần huấn luyện RL, trong khi vẫn tận dụng ngữ cảnh tìm kiếm hay không? [Paper: PDF pp. 1–2]

## 04 Bối cảnh và đường phát triển

[Paper-framed; external verification not performed] Paper đặt probability matching/MAB vào nhóm AOS ít ngữ cảnh, RL vào nhóm có trạng thái nhưng tốn huấn luyện; LAOS là một giải pháp LLM training-free. Không suy ra đây là phương pháp đầu tiên trong toàn bộ lĩnh vực. [Paper: PDF pp. 1–3]

## 05 Nút thắt mà paper nêu

[Paper] Chọn toán tử thiếu ngữ cảnh hoặc cần huấn luyện; truyền toàn bộ trajectory cho LLM cũng dư thừa. [Paper: PDF p. 5, Section 3.2]

## 06 Ý tưởng cốt lõi

[Paper] Chuyển trạng thái tìm kiếm và operator credit thành prompt có cấu trúc, bổ sung trải nghiệm tương tự từ hai tầng bộ nhớ để LLM chọn toán tử. [Analysis] Đây là chọn HOW ở cấp operator, không phải chẩn đoán vùng UCTP và không phải điều chỉnh thời điểm gọi. [Paper: PDF pp. 3–6]

## 07 Tổng quan phương pháp

[Paper] Đầu vào: quần thể, tập toán tử, trạng thái, credit và replay. Đầu ra: toán tử được chọn cho `p` thế hệ tiếp theo. Chạy fixed-operator để thu thập offline experiences; khi tối ưu, lưu online experiences; mỗi `p` vòng tính trạng thái/credit, lấy mẫu tương tự và gọi LLM. Không fine-tune LLM. [Paper: PDF p. 4, Algorithm 1]

## 08 Phân rã module

- [Paper] State vector gồm fitness trung bình/tốt nhất, progress, độ lệch chuẩn fitness, improvement, fitness range, operator hiện tại và best solution; Table 1 là nguồn định nghĩa. [Paper: PDF p. 5, Table 1]
- [Paper] Credit assignment FRRCA dùng FIR và decay; replay gồm offline/online, EES loại trùng và SMS lấy trải nghiệm tương tự. [Paper: PDF pp. 5–6, Algorithm 2]
- [Paper] Prompt nêu problem, operator pool, log và output format; parser/EA thực hiện operator. [Paper: PDF p. 4, Figure 2]

## 09 Công thức và ký hiệu thiết yếu

[Paper] `p` là khoảng cập nhật operator cố định; Equation 1 là reward decay/FRRCA và Equation 2 là trọng số SMS theo khoảng cách fitness. Các công thức này phụ thuộc fitness vô hướng của paper, không được chuyển nguyên sang NSGA-III đa mục tiêu. [Paper: PDF pp. 4–6, Equations 1–2]

## 10 Thiết kế thực nghiệm và chuỗi bằng chứng

[Paper] BBOB, TSP, JSP, WTA; đối chứng random, FRRAP, FRRMAB, DQN/PPO; 10 lần chạy/instance. Table 2 và Table 3 so sánh nghiệm; Figure 3 và Figure 4 là performance/convergence; Figure 5 ablate state/credit/replay, Figure 6 thay `p`/LLM, Figure 7 thay prompt, Table 4 đo runtime. Figure 1 và Figure 2 trình bày cơ chế và prompt; Table 1 định nghĩa feature. [Paper: PDF pp. 3–8, Figures 1–7, Tables 1–4]

## 11 Diễn giải kết luận đúng mức

[Paper] Kết quả trong các bài toán đơn mục tiêu được thử cho thấy LAOS có thể cải thiện chất lượng nghiệm nhưng chi phí LLM lớn hơn AOS công thức. [Analysis] Không có thí nghiệm UCTP, NSGA-III hoặc quyết định WHERE/WHEN thích nghi; kết quả không dự báo hiệu năng cho đề tài hiện tại. [Paper: PDF pp. 6–8, Tables 2–4]

## 12 Giới hạn tác giả thừa nhận

[Paper] Chi phí inference LLM có thể làm mất lợi thế ở bài toán có đánh giá nghiệm rẻ; hướng tương lai nêu multi-objective LAOS và trạng thái gọn hơn. [Paper: PDF p. 8, Computational Cost Analysis/Conclusions]

## 13 Phân tích phê bình

[Analysis] State `f4` là độ lệch chuẩn fitness vô hướng; NSGA-III cần một chỉ số đa mục tiêu khác. Cần kiểm tra liệu cải thiện chỉ đến từ operator history hay từ LLM bằng đối chứng có cùng thông tin. [Paper: PDF p. 5, Table 1; PDF p. 7, Figure 5]

## 14 Kiến thức có thể chuyển giao

[Analysis] Dùng summary định lượng + recent operator outcome; output có schema giới hạn; ghi runtime và ablate từng nguồn thông tin. [Paper: PDF pp. 4–8]

## 15 Liên hệ với đề tài

[Analysis] LAOS là xuất phát điểm HOW. Có thể giữ operator pool/feedback chung cho single-agent và Inspector–Planner, nhưng phải thiết kế mới vị trí can thiệp và review timing. Liên hệ này chỉ dựa trên paper và proposal local; chưa xác minh novelty ngoài bốn paper. [Paper: PDF pp. 3–5]

## 16 Ý tưởng nghiên cứu

[Hypothesis] `WHERE+HOW+WHEN` có thể hiệu quả hơn chỉ chọn operator toàn cục nếu mục tiêu/chi phí địa phương thật sự phản ánh bottleneck. Delta: thêm target và interval vào action space, giữ solver-side validation. Validation: so NSGA-III, controller một agent và hai agent trên cùng instances/seeds/wall-clock; falsifier: không cải thiện sau khi tính overhead. Failure modes: summary kém, LLM chậm, operator địa phương không phù hợp. Innovation status: unverified.
