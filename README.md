# Event-Triggered LLM Control for University Course Timetabling

Repository này phục vụ nghiên cứu các phương pháp tiến hóa cho bài toán xếp thời khóa biểu đại học (University Course Timetabling Problem, UCTP) trên bộ benchmark International Timetabling Competition 2019 (ITC 2019).

## Current status

Hiện repository cung cấp baseline NSGA-III cho ITC 2019. Bộ điều khiển LLM theo sự kiện là hướng mở rộng dự kiến và **chưa được triển khai trong phiên bản này**. Baseline vẫn đang được kiểm chứng; thành phần xếp sinh viên sử dụng heuristic nên kết quả hiện tại chưa đại diện cho hiệu năng thi đấu.

## Thành phần repository

```text
.
├── Dataset_ITC2019/       # Các instance XML gốc của ITC 2019
└── NSGA-III/               # Mã nguồn baseline và hướng dẫn chạy
```

Thư mục dữ liệu gồm 36 instance XML: 30 instance thi đấu và 6 test instance. Mỗi file XML mô tả một bài toán độc lập. Có thể chạy toàn bộ dữ liệu hoặc chọn subset bằng tùy chọn instance/manifest của baseline.

## Bắt đầu nhanh

Yêu cầu Python 3.10 trở lên. Chạy các lệnh sau từ thư mục gốc repository:

```powershell
python -m pip install -r NSGA-III/requirements.txt
cd NSGA-III
python main.py --instance wbg-fal10 --population 40 --generations 20 --seed 42
```

Hướng dẫn cài đặt chi tiết, tùy chọn lệnh, output và cách đánh giá thí nghiệm được trình bày trong [NSGA-III/README.md](NSGA-III/README.md).

## Kiểm chứng

Bộ đánh giá nội bộ báo cáo vi phạm hard và penalty có trọng số. Trước khi dùng điểm số để so sánh trong nghiên cứu, cần kiểm tra file nghiệm bằng [validator chính thức của ITC 2019](https://www.itc2019.org/validator). Heuristic xếp sinh viên thất bại không chứng minh instance không có nghiệm khả thi.

## Tài liệu tham khảo

- [ITC 2019 problem format](https://www.itc2019.org/format)
- [ITC 2019 competition paper](https://www.itc2019.org/papers/itc2019-patat2018.pdf)
