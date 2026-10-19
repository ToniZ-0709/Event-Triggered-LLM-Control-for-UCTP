# Guide: NSGA-III + Single-Agent Controller

## 1. Đường đi của một instance

1. `main.py` dùng parser và CLI của baseline để đọc một XML trong `../Dataset_ITC2019/`. Mỗi XML là một bài toán độc lập.
2. NSGA-III khởi tạo và tiến hóa quần thể với cùng chromosome, bốn objective, hard constraint và routine repair như baseline. Xem [GUIDELINE.md](../NSGA-III/GUIDELINE.md) để hiểu các phần này.
3. Callback theo dõi mỗi thế hệ. Sau warm-up, review diễn ra khi đến deadline hoặc có stagnation hay diversity thấp, với khoảng cách tối thiểu giữa hai review.
4. Code tạo `StatePacket` phiên bản 1, danh sách candidate regions và các action hợp lệ. Một lời gọi Gemini Developer API dùng JSON schema trả một `ActionPlan` gồm WHERE, HOW, WHEN. Không dùng OpenAI hoặc fallback trả phí.
5. Code kiểm tra toàn bộ ID, operator, budget và interval. Nếu action hợp lệ, operator thử trong vùng đã xác minh và đưa tối đa một candidate cải thiện qua NSGA-III survival. Invalid JSON/schema/action thành fallback no-op có reason; lỗi hạ tầng API làm dừng sớm, lưu run incomplete và exit code 3, không phải model chọn no-op.
6. Run ghi nghiệm cuối và trace. `solution.xml` chỉ xuất khi bộ kiểm tra nội bộ báo hard bằng 0.

`StatePacket` gồm tiến độ và phân bố bốn objective, số cá thể khả thi, best internal score, non-dominated count, decision diversity và placement diversity, candidate regions, outcome gần đây và action ranges. Mỗi candidate region nêu placement hiện tại, số option time/room, số class có thể đổi, số cặp swap hợp lệ và `available_operators`; action chọn operator ngoài danh sách này bị từ chối. Candidate regions được xếp hạng bằng penalty theo class và vi phạm distribution/room; các penalty sinh viên chưa quy được cho class được đánh dấu `unlocalized_penalties`. Region và boundary được sinh bằng code, không cho LLM tự tạo ID.

Ba operators chung là `move`, `swap` và `destroy_repair`. Operator cuối thử joint CP trên toàn vùng đã chọn (tối đa 3 giây), rồi fallback tuần tự tối đa 3 class nếu còn budget. Mọi trial qua full scorer; routine repair vẫn dùng chung với baseline. Khi nguồn `hard>0`, candidate chỉ cần giảm hard; khi nguồn `hard=0`, giữ hard 0 và giảm soft total. Chỉ nghiệm cuối khả thi mới xuất XML.

## 2. Cài đặt và chạy bằng Windows CMD

Từ `Experiment`, kích hoạt venv ngoài thư mục đồng bộ và cài Google SDK:

```cmd
call "%USERPROFILE%\venvs\itc2019-nsga3\Scripts\activate.bat"
python -m pip install -r "Single-Agent Controller\requirements.txt"
python tools/gemini_preflight.py
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --dry-run
```

Nếu chưa có venv, tạo bằng `python -m venv "%USERPROFILE%\venvs\itc2019-nsga3"` rồi activate. Preflight mặc định/dry-run không network và không cần key. Khi sẵn sàng, tạo key trong [AI Studio](https://aistudio.google.com/api-keys), kiểm tra project ghi **Free Tier**, không bật billing. Đặt key cục bộ trong chính CMD (thay placeholder bên dưới; không gửi key vào chat hay commit):

```cmd
set "GEMINI_API_KEY=YOUR_KEY_HERE"
python tools/gemini_preflight.py --live --confirm-free-tier
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

| Cờ | Mặc định | Tác động |
| --- | --- | --- |
| `--dataset`, `--instance`, `--manifest`, `--all` | Như baseline | Chọn XML độc lập. |
| `--population`, `--generations`, `--partitions`, `--seed` | `20`, `20`, `3`, `42` | Cấu hình CLI NSGA-III; wrapper dùng generation/evaluation/time cap theo instance. |
| `--time-limit-seconds` | Không đặt | Giới hạn wall-clock cho solver và controller trên một instance. |
| `--model` | `gemini-3.8-flash` / `GEMINI_MODEL` | Chỉ nhận model trong Free Tier allowlist; không tự đổi model. |
| `--confirm-free-tier` | Tắt | Chỉ bật sau khi kiểm tra project; có thể dùng `GEMINI_FREE_TIER_CONFIRMED=1`. Không tự xác minh billing từ key. |
| `--api-min-interval-seconds` | `10` | Khoảng cách request starts; chờ cũng tính runtime. Quota thực tế xem AI Studio. |
| `--api-max-input-characters` | `60000` | Guard cho packet + system prompt. |
| `--review-interval` | `10` | Interval mặc định và fallback. |
| `--warmup-generations`, `--stagnation-window` | `5`, `5` | Điều kiện bắt đầu và nhận biết đứng yên. |
| `--min-review-interval`, `--max-review-interval` | `3`, `20` | Miền WHEN hợp lệ. |
| `--diversity-threshold` | `0.20` | Ngưỡng theo tỷ lệ lớp có placement khác nhau giữa các lịch; tỷ lệ chromosome khác nhau vẫn được ghi log nhưng không kích hoạt review. |
| `--diversity-recovery-threshold` | `0.30` | Low-diversity event chỉ được kích hoạt lại sau khi diversity hồi phục trên mức này. |
| `--max-regions`, `--region-size` | `6`, `5` | Số vùng đề xuất và số class tối đa mỗi vùng. |
| `--max-operator-evaluations` | `24` | Budget intervention; tự giảm khi gần hết search budget. |
| `--api-timeout-seconds`, `--max-output-tokens`, `--max-api-calls` | `15`, `1024`, `100` | Wrapper đặt call cap 12. Low thinking, SDK một attempt/call, không tự retry. |
| `--max-reviews` | `12` | Số lần review tối đa mỗi instance; giống multi-agent. |

Model trong allowlist vẫn có thể bị tính phí nếu key thuộc paid project. `--confirm-free-tier` là xác nhận của người dùng, không phải bảo đảm hóa đơn. Không bật billing để test luồng này. Sau khi một instance chạy tốt, bỏ `--instance` để chạy cả sáu; nếu lỗi quota/timeout/config/network, wrapper dừng và không chạy lượt kế tiếp. Muốn đổi model phải chọn rõ, không trộn kết quả giữa các model. Transport chưa được tuning/đánh giá bằng API thật; 1.024 tokens là allowance ban đầu, gồm giới hạn của API cho thinking/output.

## 3. Output

```text
results_itc2019/run_n..._<hash>/
├── run_config.json
├── summary.json
└── <instance>/
    ├── result.json
    ├── choices.npz
    ├── anytime_trace.jsonl
    ├── controller_trace.jsonl
    └── solution.xml             # chỉ khi feasible nội bộ
```

`run_config.json` lưu solver/transport/provider/model/SDK và source hashes, không có key. `controller_trace.jsonl` ghi trigger/packet/decision, finish reason, latency/pacing, requests thực và prompt/output/thinking/total/cached tokens. `model_no_op` tách khỏi fallback; `incomplete_api_failure` và reason code ghi lỗi hạ tầng. Run lỗi vẫn lưu artifacts nhưng comparator từ chối so như một run agent hoàn chỉnh.

`runtime_seconds` gồm solver, callback và chờ API, chưa gồm đọc XML/ghi output. Scores là nội bộ. Feasible run có `expected.json` gắn hash XML; official status vẫn `not_run` đến khi Validate thật. Xem [workflow validator](../NSGA-III/GUIDELINE.md#9-đối-chiếu-với-validator-chính-thức); receipt chính thức lưu riêng, không tự công bố solution.

## 4. So sánh với các pipeline khác

Chạy cùng instance, seed, population, reference directions, generation cap và thời gian giới hạn cho cả ba phương pháp. Single-agent và multi-agent phải dùng cùng model/version, StatePacket, candidate regions, action ranges và operator budget. Hai contrast chính là single-agent so với baseline và multi-agent so với single-agent. Xem [PIPELINES.md](../PIPELINES.md) để biết giới hạn của diễn giải kết quả.

Tài liệu Google: [SDK](https://ai.google.dev/gemini-api/docs/libraries), [structured outputs](https://ai.google.dev/gemini-api/docs/structured-output), [giá/Free Tier](https://ai.google.dev/gemini-api/docs/pricing), [quota](https://ai.google.dev/gemini-api/docs/rate-limits). JSON schema không thay validation phía solver. Token limits, quota và quyền truy cập có thể đổi; kiểm tra project trước mỗi đợt chạy.
