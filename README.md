# Event-Triggered LLM Control for University Course Timetabling

Repository này phục vụ nghiên cứu các phương pháp tiến hóa cho bài toán xếp thời khóa biểu đại học (University Course Timetabling Problem, UCTP) trên bộ benchmark International Timetabling Competition 2019 (ITC 2019).

## Current status

Repository cung cấp baseline NSGA-III và hai pipeline controller chạy qua OpenAI API. Single-agent chọn WHERE, HOW, WHEN trong một lời gọi; Inspector-Planner chia quyết định thành tối đa hai lời gọi. Cả ba dùng cùng solver ITC 2019. Baseline và bộ đánh giá nội bộ vẫn cần đối chiếu official validator trước khi dùng làm kết quả nghiên cứu.

## Cấu trúc repository

```text
.
├── Dataset_ITC2019/          # Các instance XML gốc của ITC 2019
├── NSGA-III/                 # Mã solver dùng chung và baseline
├── Single-Agent Controller/ # Pipeline một lời gọi OpenAI
├── Multi-Agent Controller/  # Pipeline Inspector-Planner hai lời gọi
└── compare_pipelines.py     # Kiểm tra và so sánh ba run cùng cấu hình
```

Thư mục dữ liệu gồm 36 instance XML: 30 instance thi đấu và 6 test instance. Mỗi file XML mô tả một bài toán độc lập. Có thể chạy toàn bộ dữ liệu hoặc chọn subset bằng tùy chọn instance/manifest của baseline.

## Bắt đầu nhanh

Yêu cầu Python 3.10 trở lên. Chạy các lệnh sau từ thư mục gốc repository:

```cmd
python -m pip install -r NSGA-III/requirements.txt
cd NSGA-III
python main.py --instance wbg-fal10 --population 40 --generations 20 --seed 42
```

README của từng phương pháp: [NSGA-III baseline](NSGA-III/README.md), [Single-Agent Controller](Single-Agent%20Controller/README.md), [Multi-Agent Controller](Multi-Agent%20Controller/README.md).

Hai pipeline controller cần `OPENAI_API_KEY` và model được chỉ định khi tối ưu; `--inspect` không cần API key.

Sau khi chạy đủ ba phương pháp với cùng instance, seed và thời gian giới hạn, kiểm tra cấu hình và so sánh theo từng instance bằng lệnh:

```cmd
python compare_pipelines.py --baseline "NSGA-III/results_itc2019/RUN_ID" --single "Single-Agent Controller/results_itc2019/RUN_ID" --multi "Multi-Agent Controller/results_itc2019/RUN_ID"
```

## Kiểm chứng

Bộ đánh giá nội bộ báo cáo vi phạm hard và penalty có trọng số. Trước khi dùng điểm số để so sánh trong nghiên cứu, cần kiểm tra file nghiệm bằng [validator chính thức của ITC 2019](https://www.itc2019.org/validator). Heuristic xếp sinh viên thất bại không chứng minh instance không có nghiệm khả thi.

## Tài liệu tham khảo

- [ITC 2019 problem format](https://www.itc2019.org/format)
- [ITC 2019 competition paper](https://www.itc2019.org/papers/itc2019-patat2018.pdf)
