# NSGA-III + Single-Agent Controller

Pipeline này chạy solver ITC 2019 chung trong `../NSGA-III/`. Tại mỗi review, một lời gọi OpenAI nhận trạng thái tìm kiếm và chọn WHERE (cá thể và vùng), HOW (operator và ngân sách) và WHEN (lần review tiếp theo). Code kiểm tra quyết định, đánh giá ứng viên trên toàn instance và đưa tối đa một ứng viên qua chọn lọc NSGA-III.

README tổng ở [repository root](../README.md) mô tả cách so sánh ba phương pháp. Baseline và evaluator chung được giới thiệu trong [NSGA-III README](../NSGA-III/README.md).

## Chạy nhanh

Cần Python 3.10 trở lên. Khi chạy tối ưu, cần biến môi trường `OPENAI_API_KEY` và model được tài khoản hỗ trợ. `--inspect` chỉ đọc XML, không cần API key. Từ thư mục này:

```cmd
python -m pip install -r requirements.txt
python main.py --instance lums-sum17 --inspect
python main.py --instance lums-sum17 --model gpt-4o-mini --population 40 --generations 20 --seed 42
```

Lệnh tối ưu sẽ gọi API và có thể phát sinh chi phí. Có thể dùng `OPENAI_MODEL` thay cho `--model`. Dùng cùng model cho single-agent và hai vai của multi-agent khi so sánh. Để giới hạn thời gian trên từng instance, thêm `--time-limit-seconds 300` và đặt `--generations` đủ lớn.

## File chính

```text
Single-Agent Controller/
├── main.py
├── controller.py
├── requirements.txt
└── README.md
```

Parser, evaluator, NSGA-III, review gate, region builder, improvement operators và OpenAI adapter đều ở `../NSGA-III/itc2019/` để hai pipeline controller dùng cùng một implementation.
