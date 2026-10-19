# Guideline: baseline NSGA-III trên ITC 2019

File này hướng dẫn thao tác và đọc output. Mô tả kỹ thuật thống nhất nằm trong [PIPELINES.md](../PIPELINES.md); các thay đổi và quyết định nằm trong [IMPROVEMENT_HISTORY.md](../IMPROVEMENT_HISTORY.md).

## 1. Một XML là một bài toán độc lập

Thư mục `../Dataset_ITC2019/` hiện có 36 file XML. Mỗi file mô tả trọn vẹn một instance: chạy subset nghĩa là chọn một số file XML, không cắt bớt class hay student trong một file. Nếu chọn nhiều file, `main.py` chạy solver riêng cho từng file theo thứ tự được chọn, dùng cùng bộ tham số và seed, kết quả từng instance nằm trong thư mục con của cùng một run.

Có thể chọn đầu vào bằng `--instance NAME` (lặp lại flag để chọn nhiều file), `--manifest FILE` (một tên mỗi dòng, dòng bắt đầu bằng `#` là chú thích), hoặc `--all` (mọi file `*.xml` trong dataset, kể cả các test instance). `--inspect` chỉ parse và in kích thước instance, không tối ưu và không tạo output.

| Thành phần XML | Dữ liệu được parser giữ lại | Vai trò trong baseline |
| --- | --- | --- |
| `<problem>` | Tên, số ngày, số tuần, số slot/ngày | Quy định trục thời gian và tên instance. |
| `<optimization>` | Bốn trọng số `time`, `room`, `distribution`, `student` | Nhân với bốn penalty trước khi đưa vào mục tiêu. |
| `<rooms>` | Phòng, thời gian không khả dụng, thời gian di chuyển, capacity | Kiểm tra trùng phòng, phòng không khả dụng và xung đột sinh viên có xét di chuyển. |
| `<courses>` | `config → subpart → class`; domain thời gian/phòng, `limit`, quan hệ `parent` | Xác định class cần xếp và các lựa chọn hợp lệ cho mỗi gene; hỗ trợ gán sinh viên. |
| `<distributions>` | Loại ràng buộc, danh sách class, `required`, penalty | `required=true` đi vào hard; còn lại đi vào soft distribution. |
| `<students>` | Mỗi sinh viên yêu cầu những course nào | Decoder chọn class cho từng yêu cầu sau khi có lịch class. |

Một time option có `days` và `weeks` dạng chuỗi bit, cùng `start` và `length` tính theo **slot**, không phải chuỗi giờ `HH:MM`. Một class có thể có nhiều time option và room option; lựa chọn chỉ được lấy từ các option được liệt kê cho chính class đó.

Ví dụ, `wbg-fal10.xml` khai báo 7 ngày, 16 tuần, 288 slot/ngày và trọng số `(1, 1, 10, 10)`. Lệnh `--inspect` hiện cho thấy 150 class, 7 phòng, 21 course, 19 sinh viên và 82 distribution. `lums-sum17.xml` có 20 class, 62 phòng, 19 course và **không có sinh viên**. Vì vậy số sinh viên, số lựa chọn và ràng buộc thay đổi đáng kể giữa các instance; kích thước file XML một mình không đủ để dự đoán thời gian chạy.

## 2. Một cá thể biểu diễn gì?

Parser giữ thứ tự class trong XML. Với `n` class, chromosome là mảng số nguyên dài `2n`:

```text
[time_index[0], ..., time_index[n-1], room_index[0], ..., room_index[n-1]]
```

`time_index[i]` là vị trí của time option trong class `i`; `room_index[i]` là vị trí của room option. Cả hai dùng chỉ số bắt đầu từ `0`. Class không cần phòng dùng room index `-1`. Toàn bộ class vẫn được xếp giờ/phòng; chromosome **không trực tiếp mã hóa sinh viên thuộc section nào**.

Ví dụ class đầu tiên của `wbg-fal10` là class ID `1`. Time option thứ hai có `start=114`; room option thứ hai là phòng ID `7`. Do đó `time_index[0]=1` và `room_index[0]=1` nghĩa là xếp class này theo đúng hai option đó. ID class/phòng trong XML và chỉ số gene là hai thứ khác nhau.

Sau khi giải mã giờ/phòng, `scoring.py` gán sinh viên vào các class theo course đã đăng ký. Khi greedy/beam thất bại, `sectioning.py` thử plan sức chứa rồi gán lại cả course bị lỗi. `ScoreEngine` cache tối đa 16 timetable đã chấm để tránh tính lại; cache hit vẫn tính một logical evaluation. Chi phí chấm không chỉ là kiểm tra va chạm phòng.

## 3. Đường đi của một run

1. `main.py` xác định các XML cần chạy, kiểm tra tham số và tính mã nhận diện run từ cấu hình, nội dung XML, phiên bản code và môi trường.
2. `instance.py` parse từng XML thành `Instance`: class, domain option, distribution, student và các bảng tra ID.
3. `construction.py` tạo quần thể ban đầu bằng domain ordering, kiểm tra required pairs với class đã đặt, partial global constraints và bounded forward checking/domain propagation. `_improve_hard` tiếp tục thử move một hoặc hai class có giới hạn.
4. Với mỗi cá thể, `scoring.py` tính bốn penalty và số vi phạm hard. `PlacementProblem` truyền bốn penalty có trọng số vào `F`, còn `hard` vào một constraint `G`.
5. Trong các thế hệ tiếp theo, crossover hoán đổi cặp gene giờ/phòng theo **class**; mutation đổi option của một số class. `RoomRepair` giữ lại placement của offspring và chỉ sửa các class liên quan đến vi phạm, thay vì xếp lại toàn bộ timetable.
6. Archive chung giữ nghiệm tốt nhất theo `(hard, total)` ngay khi chấm, trước survival. Monitor dùng group CP cho required distributions, plateau search khi nhiều lỗi phòng và adaptive room CP khi hard ≤ 12. Room CP mở vùng từ 16/16 lên preset 32/32 rồi tối đa 64/64 class/options trong **cùng 8 giây**, có xét blockers và global block/break constraints. Fallback tối đa 128 full checks/2 giây; thất bại cùng cấu trúc phòng chờ 5 → 10 → 20 thế hệ. Mọi candidate được chấm lại toàn timetable. Chỉ cần giảm hard trước khả thi, không bắt lập tức đạt hard 0.
7. Kết thúc, code ghi `result.json` và `choices.npz`; chỉ ghi `solution.xml` nếu hard = 0 và kiểm tra gán sinh viên nội bộ qua được. Output được chuẩn bị trong thư mục tạm rồi công bố thành thư mục run hoàn chỉnh. Nếu chạy lại cùng cấu hình, kết quả cũ chỉ bị thay thế **sau khi lần chạy mới hoàn tất**.

Placement repair nhắm tới lỗi phòng và required distributions; missing student requests được xử lý riêng trong sectioning fallback. Nếu fallback vẫn không gán đủ, lỗi còn lại tiếp tục tính vào `hard`.

Lưu ý khi đọc profiling: lượt `RoomRepair` tự động ngay sau sampling được bỏ qua vì initial repair đã làm trong sampling. Stage `initial_sampling` gồm construction và initial repair; `offspring_repair` chỉ ghi công việc repair offspring thực tế.

## 4. Chấm điểm và đọc `hard`

Một cá thể có bốn thành phần mục tiêu, theo đúng thứ tự trong `score.weighted`:

| Vị trí | `score` thô | Từ đâu ra |
| --- | --- | --- |
| 1 | `time` | Tổng penalty của các time option đã chọn. |
| 2 | `room` | Tổng penalty của các room option đã chọn. |
| 3 | `distribution` | Penalty của các distribution không bắt buộc bị vi phạm. |
| 4 | `student` | Xung đột lịch/di chuyển giữa các class được gán cho sinh viên. |

`score.weighted` nhân từng thành phần với trọng số tương ứng trong `<optimization>`; `score.total` là tổng bốn số sau nhân. `weighted_total` mà terminal in ra chính là `score.total`. Trong NSGA-III, **bốn giá trị weighted là bốn mục tiêu riêng**; tổng chỉ dùng khi chọn một nghiệm cuối trong các nghiệm có cùng số vi phạm hard.

`score.hard` được cộng từ bốn trường trong `result.json`:

```text
hard = room_conflicts + room_unavailable
     + hard_distributions + unassigned_requests
```

- `room_conflicts`: số cặp class trùng thời gian trong cùng phòng.
- `room_unavailable`: số class được đặt vào khoảng thời gian phòng không khả dụng.
- `hard_distributions`: số đơn vị vi phạm của distribution `required=true`; một distribution có thể đóng góp nhiều hơn 1.
- `unassigned_requests`: số yêu cầu student–course mà decoder không tìm được tổ hợp class để gán.

Vì vậy `hard=5` **không tự nói được** có 5 class hay 5 sinh viên bị lỗi. Mở `result.json → score` và xem bốn trường trên để biết nguồn của tổng đó. `hard=0` nghĩa là khả thi **theo bộ kiểm tra nội bộ hiện tại**. Nếu `hard>0`, kết quả vẫn có `result.json` và `choices.npz`, nhưng không có `solution.xml`; lúc này đừng dùng `weighted_total` để kết luận lịch tốt hay xấu so với một nghiệm khả thi.

`runtime_seconds` đo solver và callback, không gồm đọc XML, tính hash hay ghi output. `evaluations` đếm các lần **yêu cầu chấm đầy đủ** trong tìm kiếm: quần thể, routine repair, monitor repair, tạo vùng và kiểm tra operator; cache hit vẫn tính vào ngân sách. `score_engine` tách logical requests, physical calls và cache hits. `evaluation_breakdown` tách routine repair khỏi bộ đếm evaluator; hai lần xác minh cuối nằm riêng ở `final_verification`, không tính vào search cap. Kiểm tra cục bộ hoặc node CP-SAT chỉ tính runtime. `anytime_trace.jsonl` ghi cả best quần thể và `best_observed_*` từ archive. `sectioning_fallback` ghi trạng thái từng course phải giải lại; plan thành công chưa có nghĩa cả timetable khả thi.

## 5. Tham số nào thực sự thay đổi điều gì?

| Tham số | Mặc định | Tác động trong code |
| --- | --- | --- |
| `--population` | `20` | Số cá thể trong quần thể; phải không nhỏ hơn số reference directions. Tăng lên không tự bảo đảm tìm kiếm tốt hơn với cùng ngân sách. |
| `--generations` | `20` | Số thế hệ trước khi dừng. Tăng lên cho thêm cơ hội tìm kiếm, nhưng không bảo đảm `hard` về 0. |
| `--time-limit-seconds` | Không đặt | Giới hạn thời gian solver theo từng instance; điều kiện nào tới trước giữa thời gian và số thế hệ sẽ dừng run. |
| `--stop-on-feasible` | Tắt | Dừng sau thế hệ đầu tiên có cá thể `hard=0`, hoặc dừng ở generation cap nếu chưa tìm được nghiệm. Không cần đặt time limit; không có time limit thì run có thể kéo dài. |
| `--partitions` | `3` | Với 4 mục tiêu: H=3 tạo 20 directions; H=4 tạo 35. Population phải đủ chứa directions. |
| `--inner-partitions` / `--inner-scale` | Không bật / `0.5` | Thêm lớp directions bên trong. Đã hỗ trợ nhưng không chọn trong preset local. |
| `--evaluations` | Không đặt | Search cap gồm repair/controller/rechecks và cả cache hits. Batch offspring cuối được thu nhỏ theo allowance còn lại, không làm tròn lên nguyên population. Final verification tách riêng. |
| `--initial-repair-steps` / `--initial-pair-trials` | `12` / `3` | Repair quần thể ban đầu. |
| `--offspring-repair-steps` / `--offspring-pair-trials` | `8` / `2` | Repair mỗi offspring; tăng 12/3 không tốt hơn trên mọi instance trong vòng xác nhận. |
| `--mutation-rate` / `--mutation-fraction` | `0.2` / `0.005` | Xác suất mutation và tỷ lệ class được đổi khi mutation xảy ra. |
| `--crossover-rate` | `0.7` | Xác suất crossover theo class, đã chọn sau tuning. |
| `--seed` | `42` | Khởi tạo quá trình ngẫu nhiên của từng instance. Đổi seed để có các lần chạy độc lập với cùng ngân sách. |
| `--instance` / `--manifest` / `--all` | Không chọn | Chọn **file XML nguyên vẹn** cần giải; nhiều file được chạy lần lượt, không ghép thành một bài toán chung. |
| `--inspect` | Tắt | In thống kê parse; không chạy NSGA-III và không tạo output. |
| `--dataset` | `../Dataset_ITC2019` | Đổi nơi tìm XML. |
| `--output` | `results_itc2019/` | Đổi thư mục gốc lưu các run. |

Các cờ `--repair-*` đổi giới hạn monitor repair chung cho cả ba pipeline; xem `--help`. `--author`, `--institution` và `--country` chỉ đi vào metadata của `solution.xml` và `run_config.json`.

Bảng trên là **CLI defaults**, không phải toàn bộ preset đã đo. `main.py` không tự đọc `profiles/pilot.json`. Để dùng preset, gọi `python ../run_pilot.py --pipeline baseline --seed 17` từ thư mục này. Preset chọn mutation 0.35/fraction 0.01; `nbi` và `yach` dùng population 12/H=2, các file khác 20/H=3. `yach` có CP allowance 15 giây/deterministic limit 2. Wrapper áp override solver và `repair_*` theo instance giống nhau ở cả ba pipeline, không phát flag trùng.

`population × generations` chỉ ước lượng số cá thể quần thể cần chấm, **không phải** tổng search evaluations vì repair cũng chấm nhiều ứng viên; batch cuối có thể nhỏ hơn population. Preset hiện dựa trên phép đo source mới ở seed 17/42, sau sàng lọc và thử thêm seed 73. Đọc `provenance.budget_status`: `measured_local_knee` là mốc hữu ích trong phạm vi đã đo; `progressing_at_evaluation_cap` là còn horizon mở; `time_censored` là bị trần thời gian ngắt. Đây chưa phải chứng minh đã hội tụ hoặc mọi instance khả thi.

Wall cap gồm hệ số an toàn, generation reserve và dự phòng tối đa 360 giây latency cho hai agent, nên không phải dự báo baseline sẽ chạy hết từng ấy thời gian. Evaluation cap được giới hạn trong quãng đã quan sát ở cả hai seed. Không có API thật nào được gọi để hiệu chỉnh preset này.

Tăng ngân sách là bước thử nghiệm hợp lý khi smoke run còn `hard>0`, nhưng không thể thay thế việc xem **loại vi phạm nào** còn lại. Chẳng hạn, nếu `unassigned_requests` chiếm phần lớn thì giới hạn của decoder gán sinh viên có thể quan trọng hơn việc tăng thêm thế hệ.

## 6. Cài đặt và chạy trên Windows CMD

### 6.1. Chuẩn bị Python và venv

Cần Python 3.10 trở lên. Mở **Command Prompt (CMD)** tại thư mục gốc repository, nơi có `Dataset_ITC2019` và `NSGA-III`, rồi chạy các lệnh sau theo thứ tự. Nếu CMD đã đứng trong `NSGA-III`, bỏ qua dòng `cd /d NSGA-III`.

```cmd
cd /d NSGA-III
python --version
set "VENV=%USERPROFILE%\venvs\itc2019-nsga3"
if not exist "%VENV%\Scripts\python.exe" py -3 -m venv "%VENV%"
"%VENV%\Scripts\python.exe" --version
"%VENV%\Scripts\python.exe" -m ensurepip --upgrade
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt
```

Venv nằm ngoài thư mục project đang đồng bộ để việc cài package nhanh và ổn định hơn. Nếu máy không nhận `py -3`, thay lệnh tạo venv bằng `python -m venv "%VENV%"`; muốn chọn đúng Python 3.11 thì dùng `py -3.11 -m venv "%VENV%"`. Sau khi mở **cửa sổ CMD mới**, chạy lại dòng `set "VENV=..."` trước khi dùng các lệnh bên dưới.

Không cần activate: gọi trực tiếp `"%VENV%\Scripts\python.exe"` luôn dùng đúng môi trường. Nếu muốn activate trong CMD hiện tại:

```cmd
call "%VENV%\Scripts\activate.bat"
python main.py --instance wbg-fal10 --inspect
deactivate
```

CMD không hiểu cú pháp PowerShell như `& "$venv\Scripts\python.exe"`. Các lệnh trong tài liệu này dành cho CMD.

### 6.2. Chọn instance và ngân sách chạy

Khuyến nghị dùng preset đã đo. Từ thư mục `NSGA-III`, lệnh đầu chỉ xem cấu hình, lệnh sau chạy cả sáu instance; mỗi instance tự nhận ngân sách trong `../profiles/pilot.json`:

```cmd
"%VENV%\Scripts\python.exe" ..\run_pilot.py --pipeline baseline --seed 17 --dry-run
"%VENV%\Scripts\python.exe" ..\run_pilot.py --pipeline baseline --seed 17
"%VENV%\Scripts\python.exe" ..\run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 42
```

Không cần tự ghi lại population, H, repair và budget khi dùng wrapper. Một instance tạo một thư mục run độc lập vì budget có thể khác nhau. Muốn tùy chỉnh ngoài preset thì dùng `main.py` như các ví dụ dưới.

Kiểm tra một XML mà không chạy tối ưu:

```cmd
"%VENV%\Scripts\python.exe" main.py --instance wbg-fal10 --inspect
```

Chạy nhanh để xác nhận pipeline, rồi chạy một instance với ngân sách lớn hơn:

```cmd
"%VENV%\Scripts\python.exe" main.py --instance wbg-fal10 --population 40 --generations 2 --seed 42
"%VENV%\Scripts\python.exe" main.py --instance wbg-fal10 --population 100 --generations 100 --seed 42
```

Chạy lại cùng tham số sẽ thay kết quả trong cùng thư mục sau khi hoàn tất. Muốn giữ **một lần thử độc lập khác**, đổi seed, ví dụ `--seed 43`.

Chạy vài instance trong cùng một lệnh:

```cmd
"%VENV%\Scripts\python.exe" main.py --instance wbg-fal10 --instance lums-sum17 --population 40 --generations 20 --seed 42
```

Muốn chạy tới khi tìm được nghiệm khả thi (hoặc chạm generation cap), không cần truyền `--time-limit-seconds`; thêm `--stop-on-feasible`:

```cmd
"%VENV%\Scripts\python.exe" main.py --manifest ..\pilot_instances.txt --population 40 --partitions 4 --generations 1000 --seed 17 --stop-on-feasible
```

Điều kiện được kiểm tra sau khi hoàn tất một thế hệ, nên thời gian thực có thể dài và solver không ngắt giữa generation. Nếu hết 1000 thế hệ vẫn chưa có cá thể `hard=0`, run kết thúc bình thường nhưng không có `solution.xml`.

Với subset dài hơn, mở Notepad ngay từ thư mục `NSGA-III`:

```cmd
notepad subset.txt
```

Nhập mỗi dòng một tên XML (có thể bỏ đuôi `.xml`), lưu file rồi đóng Notepad:

```text
wbg-fal10
lums-sum17
```

Sau đó chạy:

```cmd
"%VENV%\Scripts\python.exe" main.py --manifest subset.txt --population 40 --generations 20 --seed 42
```

Muốn xem thống kê của tất cả XML hoặc **chạy cả 36 file**, dùng lần lượt:

```cmd
"%VENV%\Scripts\python.exe" main.py --all --inspect
"%VENV%\Scripts\python.exe" main.py --all --population 40 --generations 20 --seed 42
```

`--all` bao gồm cả sáu test instance. Nếu chỉ muốn nhóm 30 competition instance hoặc subset nhóm đã chọn, hãy liệt kê đúng tên file trong manifest. Chạy toàn bộ dataset có thể rất lâu vì mỗi XML được tối ưu độc lập; không dùng `--all` cùng `--instance` hoặc `--manifest`. Nếu dataset hoặc output ở nơi khác, truyền `--dataset "đường dẫn"` hoặc `--output "đường dẫn"` và đặt đường dẫn có khoảng trắng trong dấu nháy kép.

### 6.3. Tìm kết quả và xử lý lỗi thường gặp

Dòng cuối trên CMD in `results_dir` và `overwrote_existing`. Có thể mở thư mục kết quả bằng:

```cmd
explorer results_itc2019
```

Trong thư mục run, xem `run_config.json` để xác nhận instance, seed, population và generations; xem `<instance>\result.json` để biết `hard` gồm những loại vi phạm nào. `solution.xml` chỉ có khi `hard=0` theo bộ kiểm tra nội bộ. Smoke run 2 generations còn `hard>0` không có nghĩa là lệnh chạy bị lỗi; xem mục 4 để đọc điểm.

Nếu quá trình tạo venv trước đó dừng ở `KeyboardInterrupt`, `Scripts\python.exe` có thể đã tồn tại nhưng pip chưa hoàn tất. Từ thư mục `NSGA-III` trong CMD, chạy lại:

```cmd
set "VENV=%USERPROFILE%\venvs\itc2019-nsga3"
"%VENV%\Scripts\python.exe" -m ensurepip --upgrade
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt
```

Thông báo đường dẫn project `C:` được chuyển sang `G:` là do thư mục đồng bộ trên máy hiện tại. Đặt venv tại `%USERPROFILE%\venvs\itc2019-nsga3` giúp tránh cài package vào đường dẫn đồng bộ. Nếu báo population nhỏ hơn số reference directions, tăng `--population` hoặc giảm `--partitions` (mục 5).

## 7. Output, ghi đè và khả năng kiểm chứng

Một thư mục mới có dạng `results_itc2019/run_n1_p100_g100_r4_s42_<hash>/`. Các phần `n/p/g/r/s` đọc trực tiếp được; phần hash còn phân biệt danh sách và **nội dung** XML, đầy đủ `Config`, metadata solution, phiên bản Python/NumPy/pymoo và các file nguồn chính. Cùng các yếu tố đó sẽ dùng lại một thư mục và ghi đè kết quả cũ sau khi run hoàn tất. Nếu đổi seed, dữ liệu, code hoặc môi trường, tên thư mục sẽ đổi. Dòng cuối trên terminal in `results_dir` và `overwrote_existing`.

```text
results_itc2019/run_.../
├── run_config.json       # instance, hash XML, Config, phiên bản, thời điểm hoàn tất
├── summary.json          # danh sách kết quả của mọi instance trong run
└── wbg-fal10/
    ├── result.json       # score chi tiết, feasible, evaluations, runtime
    ├── choices.npz       # hai mảng chỉ số time/room của nghiệm cuối
    ├── anytime_trace.jsonl # thống kê quần thể mỗi thế hệ
    ├── expected.json     # score và hash XML để đối chiếu validator, khi feasible
    └── solution.xml      # chỉ có khi hard = 0 và kiểm tra nội bộ qua được
```

Để đọc kết quả, mở `run_config.json` xác nhận **đã chạy đúng dữ liệu và tham số**, sau đó mở `<instance>/result.json`. Xem `feasible` và `score.hard` trước; nếu `hard>0`, đọc bốn trường vi phạm trong mục 4. Nếu `hard=0`, xem bốn thành phần `weighted`, `total` và `runtime_seconds`. Chỉ so điểm giữa các run của **cùng instance** và cùng bộ đánh giá; khi báo cáo nghiên cứu, kiểm tra `solution.xml` bằng validator chính thức của ITC 2019.

## 8. Giới hạn cụ thể của baseline hiện tại

- Greedy/beam 12 có fallback gán lại course: tối đa 4.096 bundles, 200.000 enumeration nodes và deterministic CP budget 0,5. Có thể bỏ lỡ bundle cần thiết; `unassigned_requests>0` vẫn không chứng minh instance vô nghiệm. Plan ưu tiên đủ sức chứa, chưa bảo đảm student conflict tối ưu.
- CP-SAT chỉ giải vùng hữu hạn với phần ngoài cố định; sampled options có thể bỏ lỡ cải thiện. Local CP soft objective ưu tiên time/room, chưa tối ưu trực tiếp toàn bộ soft distribution/student penalty. Cả ba pipeline dùng cùng solver và không tự dừng ở hard = 0 nếu không bật `--stop-on-feasible`.
- Parser đọc `room.capacity` để giữ nguyên dữ liệu XML, nhưng chấm điểm sectioning dùng `class.limit`. [FAQ ITC 2019](https://www.itc2019.org/faq) xác nhận không cần so số sinh viên trong class với capacity của room. Nghiệm khả thi nội bộ vẫn cần official validator trước khi dùng làm kết quả nghiên cứu vì decoder và scorer chưa được đối chiếu đầy đủ.
- Entry point baseline không gọi LLM. Hai pipeline controller ở thư mục cùng cấp dùng chung solver này để so sánh WHERE/HOW/WHEN.

Vị trí code: [main.py](main.py) chọn dữ liệu và quản lý output; [instance.py](itc2019/instance.py) parse XML; [nsga3.py](itc2019/nsga3.py) định nghĩa chromosome, operator và vòng chạy; [scoring.py](itc2019/scoring.py) chấm điểm, gán sinh viên và kiểm tra nội bộ.

## 9. Đối chiếu với validator chính thức

- Validator kiểm file nghiệm/khả thi và penalty, không giúp solver tìm nghiệm. Không cần gọi trong mỗi generation hoặc đợi tất cả instance đạt hard 0.
- Hai bộ file đã chuẩn bị trong `../solver_checks/official_validation/`: `muni-fsps-spr17_s17_750e406396b08cbc` và `nbi-spr18_s17_86a125944e83de52`. Mỗi bộ có XML và expected score/hash, lấy từ choices đã lưu, không chạy lại GA. Trạng thái official vẫn chưa kiểm.
- Đăng nhập [ITC 2019](https://www.itc2019.org/validator), chọn `solution.xml` và **Validate**. Không cần bấm **Upload solution** để công bố nghiệm. Đối chiếu total và xem log lỗi; không đánh dấu pass chỉ vì code nội bộ báo hard 0.
- Nếu lưu được JSON response chính thức, dùng lệnh dưới từ thư mục `Experiment` (thay `PATH_TO_...` bằng đường dẫn có thật):

```cmd
python tools/validate_itc2019.py import --solution "PATH_TO_SOLUTION_XML" --response "PATH_TO_OFFICIAL_JSON"
```

- Tool đọc `expected.json` cạnh XML, từ chối XML/hash/instance lệch. `OK` và điểm khác được ghi `penalty_mismatch`; `NOT-VALID` không phải pass, `NOT-SOLUTION` không phải xác nhận trên competition instance. Lưu kết quả vào receipt `official_validation.json`, không sửa lịch sử run hay tự công bố nghiệm.
- Có thể dùng `submit --solution "PATH_TO_SOLUTION_XML"` với session token của chính bạn đã đặt cục bộ trong `ITC2019_TOKEN`. Nếu chưa có token, tool trả `auth_required`, không gửi request. Không gửi key/token/mật khẩu vào chat hoặc commit lên repo. Comparator chỉ đọc receipt khớp đúng XML/score của run.
- Tài liệu chính thức: [FAQ](https://www.itc2019.org/faq), [rules](https://www.itc2019.org/rules). Các unit test của tool chỉ kiểm response/bookkeeping, không thay validator độc lập.
