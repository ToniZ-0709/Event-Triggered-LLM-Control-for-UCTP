# Guide: NSGA-III + Inspector-Planner Controller

## 1. Đường đi của một instance

1. `main.py` dùng parser và CLI của baseline để đọc một XML trong `../Dataset_ITC2019/`. Mỗi XML là một bài toán độc lập.
2. NSGA-III khởi tạo và tiến hóa quần thể với cùng chromosome, bốn objective, hard constraint và routine repair như baseline. Xem [GUIDELINE.md](../NSGA-III/GUIDELINE.md) để hiểu các phần này.
3. Callback theo dõi mỗi thế hệ. Sau warm-up, review diễn ra khi đến deadline hoặc có stagnation hay diversity thấp, với khoảng cách tối thiểu giữa hai review.
4. Code tạo `StatePacket` phiên bản 1 và candidate regions. Inspector nhận packet để chọn `solution_id`, `region_id`, `boundary_id`, hoặc không chọn target. Code kiểm tra cả ba ID.
5. Khi target hợp lệ, Planner nhận packet gốc và diagnosis của Inspector. Planner chỉ chọn operator, execution budget và next review interval; không được đổi WHERE. Tối đa hai API calls trên một review, không có vòng hội thoại sửa lỗi.
6. Gemini Developer API trả JSON/schema, không dùng OpenAI hay provider fallback. Code kiểm tra ActionPlan rồi thử operator trong vùng. Source chưa khả thi chỉ cần giảm hard; source đã khả thi phải giữ hard 0 và giảm soft total. Quyết định sai thành fallback no-op; lỗi hạ tầng API dừng sớm và lưu run incomplete, không coi là model chọn no-op.
7. Run ghi nghiệm cuối, anytime trace và controller trace. `solution.xml` chỉ xuất khi bộ kiểm tra nội bộ báo hard bằng 0.

`StatePacket`, vùng, triggers và action library dùng chung với single-agent. `destroy_repair` thử joint CP toàn vùng đã chọn, tối đa 3 giây; còn budget mới fallback tuần tự tối đa 3 class. Planner không được đổi target hoặc chọn operator ngoài `available_operators`. Missing requests chưa localize được vẫn đánh dấu rõ. Full scorer kiểm mọi trial; routine repair gốc không chạy thêm riêng cho agent.

## 2. Cài đặt và chạy bằng Windows CMD

Từ `Experiment`, dùng cùng venv với baseline. Nếu chưa có, tạo bằng `python -m venv "%USERPROFILE%\venvs\itc2019-nsga3"`:

```cmd
call "%USERPROFILE%\venvs\itc2019-nsga3\Scripts\activate.bat"
python -m pip install -r "Multi-Agent Controller\requirements.txt"
python tools/gemini_preflight.py
python run_pilot.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --dry-run
```

Các lệnh trên không gọi API. Tạo key trong [AI Studio](https://aistudio.google.com/api-keys), kiểm tra project **Free Tier**, không bật billing. Đặt key cục bộ trong CMD, không gửi vào chat hay commit. Các lệnh dưới mới gửi request thật:

```cmd
set "GEMINI_API_KEY=YOUR_KEY_HERE"
python tools/gemini_preflight.py --live --confirm-free-tier
python run_pilot.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

| Cờ | Mặc định | Tác động |
| --- | --- | --- |
| `--dataset`, `--instance`, `--manifest`, `--all` | Như baseline | Chọn XML độc lập. |
| `--population`, `--generations`, `--partitions`, `--seed` | `20`, `20`, `3`, `42` | Cấu hình CLI NSGA-III; wrapper dùng generation/evaluation/time cap theo instance. |
| `--time-limit-seconds` | Không đặt | Giới hạn wall-clock cho solver và controller trên một instance. |
| `--model` | `gemini-3.8-flash` / `GEMINI_MODEL` | Cùng model cho Inspector/Planner; chỉ nhận Free Tier allowlist, không tự đổi model. |
| `--confirm-free-tier` | Tắt | Xác nhận project ở AI Studio; hoặc `GEMINI_FREE_TIER_CONFIRMED=1`. Không tự xác minh billing từ key. |
| `--api-min-interval-seconds` | `10` | Pace cả hai roles; thời gian chờ tính runtime. |
| `--api-max-input-characters` | `60000` | Guard cho packet/prompt, kể cả Planner packet + diagnosis. |
| `--review-interval` | `10` | Interval mặc định và fallback. |
| `--warmup-generations`, `--stagnation-window` | `5`, `5` | Điều kiện bắt đầu và nhận biết đứng yên. |
| `--min-review-interval`, `--max-review-interval` | `3`, `20` | Miền WHEN hợp lệ. |
| `--diversity-threshold` | `0.20` | Ngưỡng theo tỷ lệ lớp có placement khác nhau giữa các lịch; tỷ lệ chromosome khác nhau vẫn được ghi log nhưng không kích hoạt review. |
| `--diversity-recovery-threshold` | `0.30` | Low-diversity event chỉ được kích hoạt lại sau khi diversity hồi phục trên mức này. |
| `--max-regions`, `--region-size` | `6`, `5` | Số vùng đề xuất và số class tối đa mỗi vùng. |
| `--max-operator-evaluations` | `24` | Budget intervention; tự giảm khi gần hết search budget. |
| `--api-timeout-seconds`, `--max-output-tokens`, `--max-api-calls` | `15`, `1024`, `100` | Wrapper đặt call cap 24; low thinking, một SDK attempt/call, không tự retry. |
| `--max-reviews` | `12` | Số lần review tối đa mỗi instance; giống single-agent. |

Model trong allowlist vẫn có thể tính phí khi key thuộc paid project; flag xác nhận không phải bảo đảm hóa đơn. Sau một instance chạy tốt, bỏ `--instance` để chạy cả sáu. Nếu API bị quota/billing/auth/network/timeout, run dừng và lưu `incomplete_api_failure`, exit code 3; wrapper không chạy tiếp hàng loạt. Inspector lỗi không gọi Planner. Các transport limits mới chỉ có tests offline, chưa đánh giá/tuning bằng LLM thật; allowance 1.024 tokens chứa bounded JSON và thinking theo giới hạn API.

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

`run_config.json` lưu solver, provider/model/SDK/transport và source hashes, không lưu key. `controller_trace.jsonl` ghi trigger, packet/decision, finish reason, latency/pacing và prompt/output/thinking/total/cached tokens riêng cho mỗi role. `api_calls` đếm requests thực, không tính deadline skip như một request. `model_no_op` phân biệt với fallback; lỗi hạ tầng không được comparator coi như một run agent hoàn chỉnh.

`runtime_seconds` gồm solver, callback và chờ cả hai roles, chưa gồm đọc XML/ghi output. Scores là nội bộ. Feasible run có `expected.json` gắn hash XML; official status vẫn `not_run` đến khi Validate thật. Xem [workflow validator](../NSGA-III/GUIDELINE.md#9-đối-chiếu-với-validator-chính-thức); receipt lưu riêng, không tự công bố solution.

## 4. So sánh với các pipeline khác

Chạy cùng instance, seed, population, reference directions, generation cap và thời gian giới hạn cho cả ba phương pháp. Single-agent và multi-agent dùng cùng model/version, StatePacket, candidate regions, action ranges và operator budget. Multi-agent dùng hai lượt gọi nên có thể tiến hóa ít thế hệ hơn trong cùng thời gian, và chi phí này được tính trong runtime. Hai contrast chính là single-agent so với baseline và multi-agent so với single-agent. Xem [PIPELINES.md](../PIPELINES.md) để biết giới hạn diễn giải.

Tài liệu Google: [SDK](https://ai.google.dev/gemini-api/docs/libraries), [structured outputs](https://ai.google.dev/gemini-api/docs/structured-output), [giá/Free Tier](https://ai.google.dev/gemini-api/docs/pricing), [quota](https://ai.google.dev/gemini-api/docs/rate-limits). Kiểm tra quota/project trước khi chạy; JSON schema vẫn cần validation phía solver.
