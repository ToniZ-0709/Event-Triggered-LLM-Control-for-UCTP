# NSGA-III baseline cho ITC 2019

Đây là baseline **NSGA-III thuần**, dùng để giải bài toán xếp thời khóa biểu đại học trên các instance ITC 2019. Mục tiêu trước mắt là có một bộ giải và quy trình đo lường chung để sau này so sánh công bằng với các phương pháp có LLM controller. Phiên bản trong thư mục này **chưa có single-agent hay multi-agent controller**.

README mô tả entry point hiện tại là `main.py` cùng mã trong `itc2019/`. Các file thuộc `core/`, `data/`, `optimizer/nsga3.py` và những thư mục cũ khác là phần triển khai cho bộ dữ liệu HCMUT 261; `main.py` không gọi chúng.

## 1. Dữ liệu và bài toán

Baseline dùng **bộ ITC 2019 gốc** đã đặt ở thư mục `Dataset_ITC2019`, nằm cùng cấp với thư mục `NSGA-III`. Không dùng bộ `ITC2019_post_competition_reduced`.

```text
Experiment/
├── Dataset_ITC2019/       # các instance XML gốc
│   ├── wbg-fal10.xml
│   └── ...
└── NSGA-III/
    ├── main.py
    ├── itc2019/
    └── README.md
```

Thư mục dữ liệu hiện có **36 file XML**: 30 competition instances và 6 test instances. Mỗi file mô tả một bài toán độc lập; có thể chạy một file, một subset hoặc toàn bộ thư mục mà không cần cắt nhỏ hay sửa nội dung XML. Nếu chỉ muốn đánh giá trên subset, hãy chọn tên file bằng `--instance` hoặc `--manifest`.

Một instance gồm các course, class, lựa chọn thời gian/phòng, room, yêu cầu học phần của sinh viên và các ràng buộc phân phối giữa class. Bộ giải phải chọn thời gian và phòng cho từng class, đồng thời xếp sinh viên vào các class phù hợp. **Hard constraints** là điều kiện bắt buộc để lời giải khả thi; **soft penalties** đo chất lượng của một lời giải khả thi. Cấu trúc dữ liệu và các loại ràng buộc được mô tả trên trang [ITC 2019 format](https://www.itc2019.org/format).

## 2. Baseline hoạt động như thế nào?

![Sơ đồ pipeline NSGA-III baseline](docs/nsga3-baseline-pipeline.png)

Sơ đồ có thể tạo lại bằng `python docs/make_baseline_pipeline.py` sau khi cài thêm `matplotlib` (`python -m pip install matplotlib`). Đây chỉ là dependency để vẽ hình, không cần cho việc chạy baseline. Bản vector SVG/PDF nằm trong cùng thư mục `docs/`.

1. **Đọc XML:** parser tạo mô hình instance, bao gồm domain thời gian/phòng của từng class, sinh viên và các ràng buộc.
2. **Mã hóa lời giải:** mỗi cá thể chứa hai lựa chọn cho mỗi class: một time option và một room option. Class không cần phòng dùng giá trị room đặc biệt.
3. **Khởi tạo và tìm kiếm:** tạo quần thể ban đầu, sau đó NSGA-III lặp qua crossover, mutation, repair giới hạn và chọn lọc theo reference directions. Repair tập trung vào xung đột phòng và hard distribution constraints; nó **không bảo đảm** sẽ tìm được lời giải khả thi.
4. **Đánh giá:** mỗi cá thể có bốn mục tiêu penalty đã nhân trọng số theo instance: time, room, soft distribution và student conflict. Số vi phạm hard được dùng như một constraint riêng. Việc xếp sinh viên vào class hiện dùng bộ giải mã greedy có beam nhỏ, không phải bộ giải chính xác.
5. **Chọn kết quả:** trong quần thể cuối, chọn cá thể có ít vi phạm hard nhất; nếu bằng nhau, chọn `weighted_total` thấp nhất. Chỉ xuất `solution.xml` khi số vi phạm hard bằng 0 và bước kiểm tra sectioning nội bộ thành công.

`weighted_total` là **một điểm số dùng để chọn và báo cáo lời giải cuối**, không thay thế bốn mục tiêu mà NSGA-III sử dụng trong quá trình tìm kiếm.

## 3. Chuẩn bị môi trường

Cần Python, `numpy` và `pymoo`. File `requirements.txt` còn có một số thư viện phục vụ mã cũ; cài toàn bộ file là cách đơn giản nhất để chạy project. Mở PowerShell:

```powershell
cd "C:\HCMUT\Projects\GA\Literature Review\Experiment\NSGA-III"
python -m pip install -r requirements.txt
```

Kiểm tra dữ liệu và parser trước khi tối ưu:

```powershell
python main.py --instance wbg-fal10 --inspect
python main.py --all --inspect
```

`--inspect` chỉ đọc và in thông tin instance, **không chạy NSGA-III và không tạo thư mục kết quả**. Nếu Python trên máy được gọi bằng `py` thay vì `python`, thay tiền tố lệnh tương ứng.

## 4. Chạy baseline

Chạy thử một instance với cấu hình nhỏ để kiểm tra quy trình:

```powershell
python main.py --instance wbg-fal10 --population 40 --generations 20 --seed 42
```

Chạy nhiều instance độc lập trong một lần gọi:

```powershell
python main.py --instance wbg-fal10 --instance lums-sum17 --population 40 --generations 20 --seed 42
```

Với subset do nhóm chọn, tạo file văn bản `subset.txt`, mỗi dòng ghi một tên instance (có hoặc không có `.xml`):

```text
wbg-fal10
lums-sum17
```

Sau đó chạy:

```powershell
python main.py --manifest subset.txt --population 40 --generations 20 --seed 42
```

Muốn chỉ định thư mục dữ liệu hoặc kết quả khác, dùng `--dataset` và `--output`:

```powershell
python main.py --dataset "C:\HCMUT\Projects\GA\Literature Review\Experiment\Dataset_ITC2019" --instance wbg-fal10 --output "C:\HCMUT\Projects\GA\Literature Review\Experiment\NSGA-III\results_itc2019" --population 40 --generations 20 --seed 42
```

Lệnh `python main.py --all ...` sẽ chạy **cả 36 XML hiện có**, gồm 6 test instances, và có thể tốn nhiều thời gian trên các file lớn. Để chỉ chạy tập competition hoặc subset thí nghiệm, dùng manifest. Không cần sửa code khi danh sách subset thay đổi.

Các tùy chọn quan trọng:

| Tùy chọn | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `--instance NAME` | Chọn một instance; có thể lặp lại | Không có |
| `--manifest FILE` | Danh sách instance, một tên mỗi dòng; dòng bắt đầu bằng `#` được bỏ qua | Không có |
| `--all` | Chạy mọi `*.xml` trong `--dataset` | Tắt |
| `--population N` | Số cá thể | `40` |
| `--generations N` | Số thế hệ | `20` |
| `--partitions N` | Tham số tạo reference directions cho NSGA-III | `4` |
| `--seed N` | Random seed | `42` |
| `--output DIR` | Thư mục gốc chứa kết quả | `results_itc2019/` |

`--population` phải không nhỏ hơn số reference directions được tạo từ `--partitions`; cặp mặc định `40` và `4` dùng được. Thay đổi `--partitions` có thể đòi hỏi tăng population. Chạy `python main.py --help` để xem đầy đủ tùy chọn, bao gồm metadata `--author`, `--institution` và `--country` cho XML xuất ra.

## 5. Output có gì và đọc như thế nào?

Mỗi lần chạy tối ưu tạo một thư mục mới có timestamp và seed, tránh ghi đè các lần chạy trước:

```text
results_itc2019/
└── <timestamp>_seed42/
    ├── summary.json
    ├── wbg-fal10/
    │   ├── result.json
    │   ├── choices.npz
    │   └── solution.xml       # chỉ có khi hard = 0
    └── <instance-khac>/
        ├── result.json
        └── choices.npz
```

- `summary.json`: danh sách kết quả của các instance trong lần chạy, thuận tiện để tổng hợp sau này.
- `result.json`: tên instance, kích thước bài toán, thời gian chạy, số lần đánh giá, cấu hình và điểm của lời giải cuối. `feasible: true` tương ứng với `score.hard == 0` theo bộ đánh giá **nội bộ**.
- `choices.npz`: hai mảng `time_indices` và `room_indices` của lời giải cuối; đây là chỉ số lựa chọn trong XML, **không phải** file nộp cho ITC. File này vẫn có khi nghiệm chưa khả thi.
- `solution.xml`: lịch cùng danh sách sinh viên được gán vào các class; chỉ được tạo cho lời giải mà bộ kiểm tra nội bộ cho là khả thi. Cần kiểm tra lại bằng [ITC 2019 validator chính thức](https://www.itc2019.org/validator) trước khi dùng trong báo cáo.

Trong `result.json`, `score.hard` là tổng số vi phạm hard do bộ chấm nội bộ ghi nhận; `room_conflicts`, `room_unavailable`, `hard_distributions` và `unassigned_requests` là các thành phần giúp chẩn đoán. `score.time`, `room`, `distribution`, `student` là penalty trước trọng số; `score.weighted` chứa bốn giá trị sau trọng số; `score.total` là tổng của bốn giá trị đó. Màn hình cũng in tóm tắt `hard`, `weighted_total`, `runtime_seconds` và có xuất `solution.xml` hay không.

Một lần chạy thử hiện có trên `wbg-fal10` với population 40, 20 generations và seed 42 cho `hard = 0`, `weighted_total = 315`, 800 evaluations, thời gian tối ưu khoảng 26 giây trên máy đã chạy thử. Đây chỉ là **smoke test để kiểm tra chương trình**, chưa phải kết quả benchmark đã đối chiếu bằng validator chính thức và không chứng minh baseline mạnh trên toàn bộ ITC 2019.

## 6. Đánh giá thí nghiệm cho đúng

Trước khi so sánh với single-agent hoặc multi-agent controller, nên chốt danh sách XML trong manifest, tập seed, population, generations **hoặc** ngân sách thời gian, cách tính tổng thời gian và tiêu chí dừng. Cần dùng cùng dữ liệu và ngân sách so sánh phù hợp giữa các phương pháp; không nên chỉ so một lần chạy ngẫu nhiên.

Với từng instance và seed, ghi nhận tối thiểu: có nghiệm hợp lệ theo validator chính thức hay không, số vi phạm hard, tổng penalty của nghiệm hợp lệ, số evaluations và wall time. Khi tổng hợp qua nhiều seed, báo cáo tỷ lệ tìm được nghiệm hợp lệ cùng trung vị và độ phân tán của penalty/runtime. Chỉ so penalty giữa các lời giải **hợp lệ của cùng một instance**; `weighted_total` thấp của nghiệm vi phạm hard không có nghĩa nghiệm đó tốt hơn nghiệm hợp lệ. Điểm số của hai instance khác nhau cũng không nên so trực tiếp như cùng một thang đo.

Hiện tại parser đã đọc được 36 XML trong thư mục dữ liệu và chương trình đã chạy thử trên một số instance nhỏ. Tuy nhiên, **chưa có kết quả đầy đủ trên tập competition** và bộ tính điểm nội bộ chưa được đối chiếu có hệ thống với validator chính thức. Bộ giải mã sectioning greedy có thể báo chưa khả thi dù tồn tại cách xếp sinh viên hợp lệ; repair cũng chỉ là heuristic giới hạn. Vì vậy nên xem đây là **baseline đang được kiểm chứng**, không phải solver ITC 2019 đạt chuẩn thi đấu.

## Tài liệu đối chiếu

- [ITC 2019 — problem format](https://www.itc2019.org/format)
- [ITC 2019 — official validator](https://www.itc2019.org/validator)
- [ITC 2019 competition paper](https://www.itc2019.org/papers/itc2019-patat2018.pdf)
