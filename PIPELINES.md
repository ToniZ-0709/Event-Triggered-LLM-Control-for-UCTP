# Pipeline hiện tại

Tài liệu mô tả code đang chạy, không phải danh sách dự định hoặc báo cáo số liệu. Quyết định và lịch sử nằm trong [IMPROVEMENT_HISTORY.md](IMPROVEMENT_HISTORY.md); thao tác CMD nằm trong [Guideline](NSGA-III/GUIDELINE.md).

## 1. Phần dùng chung và phần riêng

| Pipeline | Entry point | Quyết định riêng |
| --- | --- | --- |
| Baseline | `NSGA-III/main.py` | Không gọi LLM. |
| Single-agent | `Single-Agent Controller/main.py` | Một agent chọn WHERE, HOW, WHEN. |
| Inspector–Planner | `Multi-Agent Controller/main.py` | Inspector chọn WHERE; Planner chọn HOW và WHEN. |

Cả ba dùng cùng parser, representation, scorer, construction, NSGA-III và monitor repair trong `NSGA-III/itc2019/`. Baseline là **NSGA-III có repair**, không phải crossover/mutation thuần. Không dành một bản solver yếu hơn cho baseline. Hai controller dùng cùng action library, boundary validation và acceptance.

## 2. Dữ liệu, chromosome và score

- Mỗi XML ITC 2019 là một instance độc lập. Subset là danh sách XML nguyên vẹn.
- Chromosome dài `2n`: n time-option indices rồi n room-option indices. Class không cần phòng dùng `-1`.
- Student assignment không nằm trong chromosome. Scorer giải mã theo course/config/subpart/parent và class limit.
- Bốn objective là weighted time, room, soft distribution và student conflict. NSGA-III vẫn chọn lọc đa mục tiêu; tổng penalty chỉ dùng chọn nghiệm đại diện.

```text
hard = room_conflicts + room_unavailable
     + hard_distributions + unassigned_requests
```

Hard 0 là khả thi nội bộ; không phải chứng nhận từ validator độc lập.

### Student sectioning

Greedy xử lý sinh viên nhiều course trước, beam 12 để chọn bundle class. Khi còn yêu cầu chưa gán, fallback giải lại **toàn bộ người đăng ký course bị lỗi**, thay vì chỉ cố nhét sinh viên cuối:

1. Sinh các bundle hợp lệ theo config/subpart/parent.
2. CP-SAT tìm số sinh viên dùng từng bundle, bảo đảm tổng demand và class limits.
3. Phân các bundle đó cho sinh viên, ưu tiên ít xung đột với các course khác.
4. Tính lại missing requests và student conflicts từ assignment cuối.

Plan sức chứa không phụ thuộc timetable nên cache theo instance/course; ghép sinh viên theo xung đột vẫn phụ thuộc timetable. Enumeration giới hạn 4.096 bundles/200.000 nodes; CP một worker, seed cố định, deterministic budget 0,5. Nếu hết giới hạn hoặc không có plan trong tập bundle đã sinh, giữ lỗi unassigned; không kết luận instance vô nghiệm. Fallback ưu tiên khả thi sức chứa, không bảo đảm student penalty tối ưu.

## 3. Luồng solver và incumbent

```text
XML → constraint-aware construction → initial hard repair → initial population
    → crossover / mutation → local hard repair → full scoring → NSGA-III survival
    → shared distribution / room repair → incumbent và trace
    → optional controller review → validated regional operator → survival
    → generation / evaluation / time cap → verify incumbent → output
```

1. **Construction:** domain nhỏ trước, mức liên kết và seeded tie-break; kiểm tra required pairs với class đã đặt, partial global constraints, room usage và unavailable. Forward checking tối đa 4 hàng xóm, lan truyền pair-compatible domains; giao domain cho SameRoom/SameStart nếu có. Mỗi class lấy mẫu tối đa 32 time/16 room rồi giữ tối đa 64 placements đa dạng. Domain bị cắt không có nghĩa domain gốc vô nghiệm.
2. **Initial repair:** single moves, rồi bounded joint moves hai class. Lượt `RoomRepair` tự động của Pymoo sau sampling được bỏ qua để tránh repair hai lần.
3. **Variation:** crossover giữ cặp time/room theo class; mutation đổi một phần nhỏ class. Routine repair sửa quanh vi phạm, không dựng lại toàn bộ timetable.
4. **Scoring/survival:** feasibility handling và bốn objective; selection dùng RNG của solver.
5. **Archive:** mọi candidate đã full-score đều đi qua một incumbent chung trong `PlacementProblem`, trước survival. Nghiệm tốt không bị mất khi survival loại nó. Monitor, progress, controller feedback và final output cùng tham chiếu archive theo `(hard, total)`.
6. **Final:** đối chiếu archive/quần thể, xác minh lại score; không tự dừng ở hard 0 trừ khi bật `--stop-on-feasible`.

Archive này giữ một nghiệm đại diện, không phải external Pareto archive đầy đủ.

## 4. Repair chung

| Nhánh | Gate | Cách làm và budget |
| --- | --- | --- |
| Routine repair | Sau construction/variation | Single/two-class moves; xét exhaustive theo product domain ≤ 96, domain lớn lấy mẫu. Kiểm tra deadline bên trong vòng lặp. |
| Distribution group CP | Mốc 5 generations, chưa khả thi, distribution violations không ít hơn room clashes | Seed từ required group đang vi phạm; sửa đồng thời nhiều class với boundary cố định. Dùng deadline chung 5 giây; còn thời gian mới thử tối đa 3 rounds pair fallback, 8 pairs/round. |
| Room plateau | Mốc 5 generations, hard > 12, lỗi phòng ít nhất 4 | Local deltas và neutral/worse intermediate states; tối đa 5 giây/16 full checks, chỉ trả cải thiện cuối. |
| Adaptive room CP | Lỗi phòng còn tồn tại, hard ≤ 12, qua interval/backoff | Từ vùng 16 class/16 options lên preset 32/32, cuối cùng tối đa 64/64 nếu cần. **Cùng một** deadline 8 giây/check budget, không phải 8 giây cho mỗi tầng. |
| Beam fallback | Room CP chưa cải thiện, còn budget | Tối đa 2 giây/128 full checks. Không đưa intermediate state vào population. |
| Failure backoff | Cấu trúc lỗi phòng đã thử thất bại | Hoãn 5 → 10 → 20 generations; thay đổi cấu trúc lỗi/hard có thể mở target mới. |

Các số trên là mặc định; `RepairSettings`/CLI/profile điều khiển các giới hạn tương ứng. Vùng room mở sang lớp đang chặn lựa chọn thay thế, kể cả lớp hiện không vi phạm.

CP mã hóa room clash/unavailability, pairwise required distributions, MaxDays, MaxDayLoad, MaxBreaks và MaxBlock, kể cả tương tác với phần ngoài vùng. Block models giữ số lượng class trùng interval, xử lý singleton và gom week/day patterns giống nhau. Model quá lớn hoặc hết thời gian thì trả trạng thái budget, không kết luận infeasible toàn bài toán. Full scorer luôn kiểm lại cả timetable và sectioning.

Room branch yêu cầu giảm tổng hard **và** lỗi phòng; distribution branch chỉ cần giảm tổng hard, không đòi có lỗi phòng. Candidate cuối vẫn qua survival; archive giữ nghiệm tốt trước bước đó.

## 5. Controller: WHERE, HOW, WHEN

### WHEN

Warm-up generation 5; khoảng cách tối thiểu 3; checkpoint mặc định 10. Review khi đến checkpoint, stagnation 5 generations hoặc placement diversity ≤ 0,20. Event low-diversity rearm khi diversity > 0,30; stagnation chỉ rearm khi có tiến bộ. Agent đặt interval trong [3, 20]. Progress dùng incumbent: hard trước khả thi, total sau khả thi.

Mặc định tối đa 12 reviews. Profile cấp single tối đa 12 calls, multi tối đa 24 calls; timeout 15 giây/call. Adapter hiện dùng Gemini Developer API, model mặc định `gemini-3.8-flash`, low thinking và tối đa 1.024 output tokens (allowance kỹ thuật mới, chưa tuning bằng API thật). Kiểm tra budget trước review/operator. Không gọi API ở baseline.

Chỉ gửi request sau khi có `GEMINI_API_KEY` và xác nhận project Free Tier qua `--confirm-free-tier` hoặc `GEMINI_FREE_TIER_CONFIRMED=1`. Code không bật billing, không dùng OpenAI và không fallback provider/model. Key không cung cấp API tra billing tier, nên xác nhận của người dùng không phải bảo đảm kỹ thuật về hóa đơn. Allowlist model có Free Tier cũng không biến key thuộc paid project thành miễn phí.

Pacing mặc định 10 giây giữa các request starts, SDK attempts = 1, input cap 60.000 ký tự; tắt cả automatic continuation và automatic function calling để SDK không tự sinh calls ngoài sổ đếm. Pacing/latency tính vào runtime/deadline; không bảo đảm quota thực tế của mọi project. Đếm requests thực, prompt/output/thinking/total/cached tokens riêng. Lỗi quota, billing, cấu hình, network hoặc timeout được ghi là lỗi hạ tầng; dừng sớm run, lưu `incomplete_api_failure`, exit code 3. Wrapper không tiếp tục hàng loạt sau lỗi; comparator từ chối run lỗi hoặc không có API calls, không diễn giải chúng thành agent chọn no-op.

### WHERE và StatePacket

- Tối đa hai representatives: nghiệm tốt nhất theo hard/total và một chromosome khác, ưu tiên objective/placement trade-off khi hard tương đương. Nếu không có ứng viên gần tương đương, giữ đại diện còn lại theo thứ tự hard/total.
- Tối đa 6 connected regions, mỗi region mặc định 5 class.
- Pairwise violation quy về đúng hai đầu đang vi phạm; global penalty chia heuristic trong nhóm, không diễn giải như attribution chính xác.
- Student conflicts được localize từ assignment thực của scorer. Missing course requests vẫn đánh dấu unlocalized.
- Boundary gồm distribution relationships (kể cả đang thỏa) và các lớp chiếm alternative rooms. Packet chỉ hiện tối đa 32 boundary IDs, kèm tổng count/truncated; nội bộ vẫn giữ đầy đủ boundary và full scoring.

Packet chứa global progress/diversity, regions/current placements/allowed moves, 5 outcomes gần nhất và action limits. Incumbent và population metrics được phân biệt; outcome progress không phụ thuộc candidate có sống sót trong population hay không.

### HOW và execution

Single-agent quyết định cả ba phần trong một call. Inspector chọn region trước; solver xác minh target rồi Planner chọn operator/budget/interval. Agent chỉ chọn target/operator có sẵn, không chạy mã tự sinh.

| Action | Thực hiện |
| --- | --- |
| `move` | Đổi time hoặc room của một class trong vùng. |
| `swap` | Hoán đổi hai placements tương thích với domain của từng class. |
| `destroy_repair` | CP sửa **đồng thời toàn vùng đã chọn**, tối đa 3 giây, fixed outside; dùng phần evaluation budget còn lại cho fallback tuần tự tối đa 3 class. |
| `no_op` | Không sửa, vẫn có next review interval hợp lệ. |

Action/IDs/boundary/operator/budget/interval phải qua validation. Khi source chưa khả thi, candidate phải giảm hard; khi đã khả thi, giữ hard 0 và giảm total. Không bắt candidate chưa khả thi phải đạt ngay hard 0. Operator trials/recheck cùng tính vào search budget. JSON/schema/action sai thành fallback no-op có reason; phân biệt `model_no_op` với fallback và lỗi hạ tầng, không gộp chúng thành chất lượng lựa chọn của LLM.

CP soft branch ưu tiên weighted time/room; student/soft distribution được kiểm bằng full scorer, chưa được tối ưu trực tiếp trong local CP objective.

## 6. Scoring, budget và setting

`ScoreEngine` là LRU cache theo instance và toàn chromosome, tối đa 16 entries, giữ score cùng assignment phục vụ region. Cache hit vẫn tiêu **logical evaluation** để tránh lợi thế đếm budget cho pipeline lặp lại candidate.

`evaluations = Pymoo/monitor/controller checks + routine repair checks`; đối chiếu được với `score_engine.logical_requests`. Physical calls/cache hits ghi riêng; local delta và CP node chỉ tính runtime. Hai final verification calls liệt kê riêng, không cộng vào search budget.

Evaluation cap được giữ bằng cách reserve objective checks trong repair và thu nhỏ batch offspring cuối theo ngân sách còn lại. Không làm tròn lên nguyên population rồi vượt cap; cache hit vẫn tính vào allowance đó.

Deadline được truyền xuống construction/routine/CP/plateau/operator. Không ngắt giữa một full score; initial population hoặc scoring đã bắt đầu vẫn có thể hoàn tất sau time cap. Time limit là hàng rào hợp tác, không phải hard process timeout.

| Solver setting | Mặc định CLI |
| --- | --- |
| Population / H | 20 / 3 (20 directions) |
| Crossover / mutation rate / mutation fraction | 0,7 / 0,2 / 0,005 |
| Initial repair steps / pair trials | 12 / 3 |
| Offspring repair steps / pair trials | 8 / 2 |
| Inner reference layer | Tắt; tùy chọn `--inner-partitions` |
| Generations / seed khi gọi main trực tiếp | 20 / 42 |
| Stop-on-feasible | Tắt |

Các mặc định CLI trên không tự đọc JSON. Để dùng cấu hình đã đo, chạy `run_pilot.py` với [profiles/pilot.json](profiles/pilot.json); wrapper dùng seed 17 nếu không truyền seed.

Preset hiện chọn mutation rate **0,35**, class fraction **0,01**. `nbi-spr18` và `yach-fal17` dùng population 12/H=2; bốn instance còn lại giữ 20/H=3. Crossover, initial repair, offspring repair và inner-layer settings giữ như bảng. Riêng `yach-fal17` dùng room-CP deadline 15 giây/deterministic limit 2; các instance khác giữ 8 giây/0,5. Đây là lựa chọn trong phạm vi đã thử, không phải tham số tối ưu toàn dataset.

Wrapper áp override solver/`repair_*` theo instance cho **cả ba pipeline**, mỗi flag chỉ xuất hiện một lần. Các pipeline phải dùng cùng profile và seed khi so cùng instance; không bắt buộc các instance khác nhau có cùng population hay budget.

Caps đã được đo trên source hiện tại với seed 17/42, sau sàng lọc và thử thêm seed 73. Xem `provenance.budget_status`: `measured_local_knee` là mốc hữu ích trong khoảng đã đo; `progressing_at_evaluation_cap` nghĩa chưa quan sát đủ đuôi tiến triển; `time_censored` nghĩa trần thời gian ngắt trước horizon mong muốn. `muni-fi`, `yach`, `bet` vẫn có horizon mở; `recalibration_required` được giữ đúng trạng thái này. Không suy ra chạy thêm sẽ vô ích.

Evaluation cap không vượt quãng đã quan sát ở cả hai seed. Wall cap có hệ số an toàn, dự phòng generation dài và tối đa 360 giây API latency cho hai agent; **không phải dự báo thời gian baseline**, cũng không phải phép đo agent thật. Thời gian thu từ các job kỹ thuật chạy đồng thời, không dùng để tuyên bố ưu thế tốc độ.

## 7. Chạy và output

Từ thư mục Experiment, sau khi cài requirements trong venv:

```cmd
python run_pilot.py --pipeline baseline --seed 17 --dry-run
python run_pilot.py --pipeline baseline --seed 17
python run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 42
```

Đổi `--pipeline` thành single/multi và cung cấp Gemini key/project Free Tier để dùng controller. Chạy `python tools/gemini_preflight.py` kiểm tra offline trước; `--live --confirm-free-tier` mới gửi một packet tổng hợp qua API. Khi chạy controller thật, thêm `--confirm-free-tier`; dry-run và inspect không gửi request, không cần key.

Mỗi run có `run_config.json`, `summary.json`; mỗi instance có `result.json`, `choices.npz`, `anytime_trace.jsonl`. Agent thêm `controller_trace.jsonl`. Result ghi score, evaluation breakdown, physical/logical scoring, sectioning fallback status, stage timing và repair events. `solution.xml` và `expected.json` (score/hash phục vụ validation) chỉ xuất khi hard 0 và sectioning validation nội bộ qua. Metadata agent ghi provider, SDK version, transport policy, model requested/returned và trạng thái API; không ghi key.

Run ID gồm setting, XML hash, source hashes và dependency versions. Cùng định danh chỉ thay output cũ sau khi run mới hoàn tất qua staging. Không trộn run của source/config khác nhau.

## 8. Trạng thái sau cải thiện và hiệu chỉnh

R01–R07 đã có implementation và regression tương ứng: shared incumbent, group CP, forward-checking construction, adaptive/global CP, scoring cache/guards, regional diagnosis/joint operator, sectioning reassignment.

Đã sàng lọc cấu hình, thử tổ hợp mutation/population, kiểm tra budget CP riêng và chạy xác nhận cả sáu instance ở hai seed. Core solver được giữ nguyên trong đợt đo; thay đổi nằm ở preset, runner, công cụ đo và kiểm tra. Các quyết định nằm tại H29–H30 trong lịch sử; records được chọn nằm ở `solver_checks/recalibration_v4/_accepted_records.json`, không tạo báo cáo Markdown số liệu riêng.

Trong phạm vi đã đo, `muni-fi`, `muni-fsps`, `mary`, `nbi` đạt khả thi nội bộ ở cả hai seed xác nhận. `yach` vẫn còn room clashes; `bet` còn cả room clashes/unavailability lẫn required distributions. Missing student requests không phải thành phần gây hard còn lại trong các endpoint này. Không coi việc thêm generations là lời giải đã được chứng minh cho hai ca khó.

Repair còn có giới hạn chính sách: room CP chỉ chạy gần khả thi; coordinated CP đòi distribution violations không ít hơn room clashes. Trạng thái lỗi hỗn hợp có thể chủ yếu rơi vào plateau/routine repair. Thử gọi CP sớm hơn trên `bet` không cải thiện ca đã đo; ép riêng các CP neighborhoods trên incumbent cũng chưa tìm được candidate trong giới hạn thử. `UNKNOWN` không phải chứng minh vô nghiệm. Cần cải thiện hiệu quả neighborhood/model và phối hợp repair trước khi khẳng định solver đã ổn định trên cả sáu instance.

Kiểm nghiệm validator độc lập và quyết định từ agent thật vẫn chưa thực hiện. Gemini adapter đã có kiểm thử offline/HTTP giả lập, chưa gọi model thật do người dùng sẽ thêm key sau. Hai XML từ saved feasible choices của `muni-fsps-spr17` và `nbi-spr18` đã được re-score/export để người dùng Validate sau khi đăng nhập; đây không phải hai lượt GA mới hay kết quả official đã pass. Tool `tools/validate_itc2019.py` chuẩn bị, submit có authentication hoặc import response, giữ raw response và đối chiếu đúng XML hash/instance/assigned classes/total. Không gọi endpoint công bố nghiệm. Comparator đọc receipt sidecar, từ chối hash/score lệch và giữ nguyên `not_run` khi chưa có bằng chứng.

Không gọi API trong các lượt calibration H29-H30. Hash trong provenance chỉ snapshot đã đo, không gán hash mới của adapter/logging với kết quả cũ. Search/repair parameters giữ nguyên; các horizon mở vẫn ghi rõ trong profile, không gọi endpoint bị time cap là hội tụ. Hai PDF đã xuất trước là snapshot H01-H30; tài liệu hiện tại tiếp tục theo H31-H32.

Solver vẫn heuristic và bounded: sampled domains/neighborhoods có thể bỏ lỡ nghiệm, sectioning fallback không bảo đảm tìm đủ bundles, chưa có Pareto archive ngoài quần thể và local CP chưa tối ưu trực tiếp mọi soft objective. Không có cơ sở khẳng định solver đã tối ưu toàn cục hay giải được mọi instance.

## 9. Bản đồ code

| Thành phần | File |
| --- | --- |
| Parse / dữ liệu | [instance.py](NSGA-III/itc2019/instance.py) |
| Construction | [construction.py](NSGA-III/itc2019/construction.py) |
| Score / sectioning fallback | [scoring.py](NSGA-III/itc2019/scoring.py), [sectioning.py](NSGA-III/itc2019/sectioning.py) |
| NSGA-III / archive / routine repair | [nsga3.py](NSGA-III/itc2019/nsga3.py) |
| Monitor / review / validation | [controller.py](NSGA-III/itc2019/controller.py) |
| CP / repair policy / room deltas | [neighborhood.py](NSGA-III/itc2019/neighborhood.py), [repair_policy.py](NSGA-III/itc2019/repair_policy.py), [room_search.py](NSGA-III/itc2019/room_search.py) |
| Regions / actions / API | [regions.py](NSGA-III/itc2019/regions.py), [operators.py](NSGA-III/itc2019/operators.py), [llm.py](NSGA-III/itc2019/llm.py) |
| Run / ghép kết quả | [run_pilot.py](run_pilot.py), [compare_pipelines.py](compare_pipelines.py) |
| Calibration / replay / budget analysis | [recalibrate_solver.py](solver_checks/recalibrate_solver.py), [analyze_recalibration.py](solver_checks/analyze_recalibration.py) |
