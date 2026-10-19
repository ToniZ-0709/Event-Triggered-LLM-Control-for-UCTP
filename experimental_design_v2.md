# Bản thiết kế thực nghiệm v2 — NSGA-III với single-agent và Inspector–Planner

**Trạng thái:** bản thiết kế để nhóm thảo luận/chốt, chưa phải giao thức đã triển khai. Đối chiếu [proposal hiện tại](Report%20Literature%20Review.pdf), mục 3, và bốn paper local trong `../Papers/Phase 2/`. Những giá trị ký hiệu và `AUTHOR_INPUT_NEEDED` phải được quyết định bằng pilot trên tập development rồi khóa trước khi chạy test. Không sửa proposal gốc trong bản này.

## 1. Câu hỏi và ranh giới so sánh

| Thí nghiệm | So sánh chính | Câu hỏi có thể trả lời | Không thể tự kết luận |
|---|---|---|---|
| E1 — single-agent | NSGA-III thuần vs NSGA-III + một LLM controller | Gói controller gồm chọn vùng, operator và lịch review có mang lại chất lượng tốt hơn trong cùng thời gian thực không? | Lợi ích là do riêng LLM: controller còn được cấp thêm improvement operators. |
| E2 — multi-agent | Single-agent vs Inspector–Planner | Phân vai WHERE và HOW+WHEN có đáng chi phí lời gọi LLM thứ hai không, khi quyền hành động và hạ tầng giữ nguyên? | MOTIF/DRAGON đã chứng minh hai vai này tốt hơn; cả hai paper không thử tình huống đó. |

Giữ NSGA-III thuần làm mốc hiển thị trong cả hai bảng kết quả, nhưng **E2 chỉ có một contrast chính là multi-agent so với single-agent**. Không thêm phương pháp thứ tư vào thí nghiệm lõi. Nếu sau này muốn quy kết lợi ích cho LLM thay vì chỉ cho thêm repair operator, cần một đối chứng không-LLM dùng cùng tập operator; đó là bước mở rộng, không phải kết luận của E1 hiện tại.

## 2. Hợp đồng chung của ba phương pháp

Tất cả dùng cùng mã NSGA-III, biểu diễn thời khóa biểu, objective vector, official evaluator, feasibility checker, `routine repair`, population/reference directions, khởi tạo, GA variation/selection và cùng bộ instance/seed. Chỉ hai phương pháp controller được gọi các `improvement operators` bổ sung; chúng dùng **cùng** danh mục, tham số hợp lệ và giới hạn thực thi. Không thay variation operators gốc bằng LLM để tránh nhầm câu hỏi nghiên cứu. [Proposal, PDF pp. 6–8; P005, PDF pp. 4–7]

Đơn vị chạy là một cặp `(instance, random seed)` với một wall-clock limit `T`. Tính vào `T` toàn bộ summary extraction, clustering, API latency, parsing/validation, improvement operator, objective/feasibility evaluation và NSGA-III. Chạy một job tại một thời điểm trên cùng cấu hình phần cứng hoặc kiểm soát tài nguyên tương đương. Ghi model/version, tham số sinh, prompt hash, thời gian từng lời gọi, token đầu vào/đầu ra và lỗi API. Kết quả online không được lấy từ cache/replay của lượt test khác.

`AUTHOR_INPUT_NEEDED`: mục tiêu NSGA-III là những thành phần nào của ITC 2019? Hàm official score nào được dùng để đối chiếu benchmark? Các file dataset chưa có trong workspace; phải xác minh format/constraint/evaluator trước khi cài bộ tạo vùng và thư viện operator. Nên tách theo **file instance độc lập**, không cắt ngẫu nhiên các event bên trong một instance vì sẽ đổi mạng ràng buộc. Chia development/test theo *instance*, khóa prompt/ngưỡng trên development; không dùng test để chọn tham số.

## 3. Hợp đồng dữ liệu chung tại mỗi review

Một hàm **thuần code** và deterministic `summarize(population, history, instance)` tạo cùng một `StatePacket` cho E1 và E2. LLM không được tự nhìn toàn bộ timetable và không tự tạo các thống kê từ văn bản thô. Packet có phiên bản schema cố định:

1. `global`: generation, elapsed/remaining time, số nghiệm hợp lệ, best feasible official score (nếu có), phân bố từng objective, tiến độ quality trong cửa sổ `W`, số nghiệm non-dominated và độ phủ reference directions. Không dùng độ lệch chuẩn của một scalar fitness như LAOS một cách nguyên xi cho NSGA-III; Table 1 của LAOS là đơn mục tiêu. [P005, PDF p. 5, Table 1]
2. `candidate_regions`: danh sách `RegionCandidate` được code trích xuất từ một số cá thể đại diện của quần thể. Mỗi mục có `solution_id`, `region_id`, tập event/lecture ID, các constraint/soft-penalty contributor có thể quy về vùng, độ lớn vùng và `boundary_id`. `solution_id`/`region_id` là ID hợp lệ để agent chọn, không phải mô tả tự do. Các ràng buộc/chi phí không phân bổ cục bộ được gắn cờ và vẫn phải kiểm tra trên nghiệm toàn cục. Ý tưởng active/static + boundary đến từ DRAGON, nhưng cách dựng cụm UCTP là thiết kế mới của nhóm. [P012, PDF pp. 3–5]
3. `recent_outcomes`: tối đa `H` bản ghi gần nhất gồm target/operator, số lần thử, validity, thay đổi objective/official score, runtime và lý do không áp dụng. LAOS cho thấy state + operator credit/history hữu ích cho chọn operator, nhưng reward đa mục tiêu phải định nghĩa lại. [P005, PDF pp. 4–6]
4. `allowed_actions`: ID của các operator và phạm vi `budget`, `review_interval` hợp lệ. Cùng tập này xuất hiện ở hai pipeline.

`RegionCandidate` được tạo trước bằng một quy tắc cố định (ví dụ: đồ thị liên thuộc constraint trên những event đóng góp nhiều vào penalty); candidate list và cách xếp hạng phải được khóa trên development. Nếu hard violations luôn bằng 0 sau `routine repair`, vùng cần dựa trên **soft-cost bottlenecks**, không chỉ hard-conflict cluster. Đây là chỗ cần thử nghiệm kỹ thuật: nếu không thể quy penalty cho vùng, không giả vờ rằng LLM đã “chẩn đoán” vùng đó.

**Hợp đồng hành động chung:** `ActionPlan = {solution_id, region_id, boundary_id, operator_id | no_op, execution_budget, next_review_interval}`. Agent có thể trả `no_op`; mọi ID/phạm vi ngoài danh sách bị từ chối. Giải thích tự do (nếu có) chỉ để audit, không được thực thi. Không cho LLM xuất mã hoặc thời khóa biểu mới. Việc dùng structured output và solver-side execution phù hợp tinh thần LAOS/DRAGON. [P005, PDF p. 4, Figure 2; P012, PDF pp. 4–5]

## 4. Review gate — cùng một cơ chế cho E1 và E2

Sau warm-up `g0`, NSGA-III chạy độc lập giữa các review. Một review diễn ra khi đến deadline `g_due` **hoặc** có một event code đo được: stagnation trong cửa sổ `W` hoặc diversity xuống dưới ngưỡng `d`. Sau review ở thế hệ `g`, controller trả `Δg`; code đặt `g_due = g + clamp(Δg, Δmin, Δmax)`. Event có thể gọi sớm, nhưng phải qua `Δmin` từ lần review gần nhất để không gọi lặp liên tục. Nếu event và deadline trùng nhau chỉ gọi một lần. Nếu LLM lỗi hoặc trả interval không hợp lệ, code đặt một interval mặc định đã khóa trên development. Quy tắc này là **đề xuất vận hành**, không phải thông số của paper nào.

`stagnation` phải gắn với một quality signal định nghĩa trước cho bài đa mục tiêu (ví dụ: archive hypervolume theo reference point cố định *hoặc* official best feasible score, và ghi rõ lựa chọn); `diversity` nên dựa trên độ phủ reference directions/decision-space diversity thay vì scalar fitness spread. Không tune ngưỡng trên test. LLM-LNS chỉ dùng quy tắc stagnation để đổi *kích thước neighborhood*, không đưa ra adaptive next-LLM-call; proposal của nhóm mở rộng theo hướng khác. [P002, PDF pp. 6, 47–48]

## 5. E1 — NSGA-III + single-agent

Tại mỗi review, một LLM nhận `StatePacket`, mô tả ngắn về objective và schema. Nó trả **một** `ActionPlan` gồm WHERE (`solution_id`, `region_id`, `boundary_id`), HOW (`operator_id`, `execution_budget`) và WHEN (`next_review_interval`). Cùng một model/prompt phiên bản cố định được dùng trong tất cả runs E1. Không tạo ngầm memory riêng ngoài `recent_outcomes` chung.

Trình tự một review:

```text
code kiểm tra trigger → tạo StatePacket và candidate list
→ gọi một LLM → parse + kiểm tra ActionPlan
→ nếu no_op/lỗi: log, hẹn review tiếp, NSGA-III chạy tiếp
→ nếu hợp lệ: sao chép cá thể đích, chạy operator trong boundary/budget
→ kiểm tra lại toàn bộ hard constraints và objective
→ ứng viên hợp lệ vào offspring pool để NSGA-III selection thông thường;
   ứng viên không hợp lệ bị loại, quần thể hiện tại không bị thay đổi
→ log outcome/latency → tiếp tục NSGA-III
```

`routine repair` vẫn chạy đúng như baseline tại các điểm vốn có trong GA. Improvement operator không được “cứu” bằng một repair bổ sung chỉ cho controller nếu baseline không có bước tương đương. Có thể phác thảo danh mục ban đầu là `move`, `swap`, `destroy–repair` cục bộ, nhưng mỗi operator phải có precondition, target format, cách đếm evaluation/time và bảo toàn feasibility theo đúng định dạng ITC 2019 trước khi khóa danh mục. Không tuyên bố ba operator này đã tương thích dataset.

Để ranh giới tích hợp không mơ hồ, một lần `apply` chỉ trả **tối đa một ứng viên** sau khi tiêu tối đa budget được cấp; operator có thể khảo sát nhiều nước đi nội bộ nhưng không bơm trực tiếp nhiều nghiệm vào quần thể. Ứng viên hợp lệ không cần cải thiện mọi mục tiêu: nó đi qua selection thông thường, vốn quyết định có giữ lại hay không.

**E1 kiểm tra:** so final feasible rate, best feasible official score tại `T`, quality đa mục tiêu và anytime curve với NSGA-III thuần. Báo thêm số review, số `no_op`, số intervention hợp lệ và phần thời gian dành cho LLM. Nếu E1 tốt hơn, kết luận chỉ là **controller package** tốt hơn baseline trong giao thức này; chưa cô lập hiệu ứng WHERE, HOW, WHEN hoặc hiệu ứng LLM.

## 6. E2 — NSGA-III + Inspector–Planner

Giữ nguyên review gate, `StatePacket`, `RegionCandidate`, `ActionPlan`, operator library, execution budget và solver-side validation của E1. Dùng **cùng model family/version** cho cả hai vai và E1 (trừ khi nhóm cố ý kiểm tra model khác trong một thí nghiệm riêng). Cấu trúc hai lời gọi:

```text
code kiểm tra trigger → tạo cùng StatePacket
→ Inspector nhận StatePacket, chọn WHERE:
   {solution_id, region_id, boundary_id, relevant_constraint_ids}
→ code kiểm tra diagnosis/ID; nếu sai hoặc no_target thì no_op và log
→ Planner nhận StatePacket gốc + diagnosis đã kiểm tra +
   allowed_operators + recent_outcomes; chọn HOW + WHEN:
   {operator_id | no_op, execution_budget, next_review_interval}
→ code ghép thành ActionPlan, xác thực và thực thi y hệt E1
→ ghi outcome để cả hai vai đều thấy trong StatePacket lần sau;
   ghi riêng latency/token của Inspector và Planner
```

Planner **cũng được xem StatePacket gốc**: proposal nói hai phương pháp có cùng search-state information, nhưng danh sách đầu vào Planner trong mục 3 hiện chưa nói rõ việc này. Nếu chỉ cấp diagnosis, E2 sẽ đồng thời thay đổi kiến trúc *và* lượng thông tin của người quyết định HOW/WHEN. Đề xuất này cần nhóm xác nhận. Inspector không được tự chọn operator/interval; Planner không được đổi target của Inspector. Nếu diagnosis không hợp lệ hoặc Planner không có operator phù hợp, dùng `no_op`/fallback chung; không mở hội thoại lặp thêm trong thí nghiệm lõi. Cách này giữ **đúng hai agent, tối đa hai lời gọi cho một review**. Có thể thử retry sau, nhưng phải là thí nghiệm riêng vì nó thay đổi ngân sách.

**E2 kiểm tra:** primary comparison với E1 ở **cùng wall-clock T**. Lời gọi thứ hai làm giảm thời gian NSGA-III tiến hóa; đó là chi phí thật của phương pháp, không được bỏ khỏi phép đo. Báo thêm tổng token/call count, generations completed, chất lượng chẩn đoán vùng (nếu có nhãn/heuristic đo độc lập), tỷ lệ kế hoạch hợp lệ, cải thiện trên mỗi intervention và phân bố `next_review_interval`. Một phân tích phụ có thể so ở cùng số review hoặc cùng số operator evaluations để hiểu chất lượng quyết định, nhưng không dùng nó thay kết luận wall-clock.

E2 kiểm định **toàn bộ pipeline hai lời gọi**, không cô lập riêng tác dụng “chuyên môn hóa vai”. Nếu multi-agent thắng, thêm một lượt suy luận/token cũng có thể là nguyên nhân. Muốn quy kết lợi ích riêng cho phân vai thì về sau cần đối chứng một agent được hai lượt gọi tuần tự với cùng tổng token budget; đối chứng đó nằm ngoài hai thí nghiệm lõi mà nhóm hiện muốn làm.

MOTIF chỉ cho tiền lệ về đánh giá tương tác multi-agent trong thiết kế solver offline; kết quả của nó không chứng minh cách tách Inspector–Planner online. DRAGON cung cấp cách biểu diễn active region/boundary nhưng Reconstructor của paper trực tiếp giải nghiệm con, khác Planner của nhóm. [P047, PDF pp. 3–7; P012, PDF pp. 3–7]

## 7. Giao thức đo và phân tích

- **Primary endpoint nên khóa trước test:** (i) tỷ lệ tìm được nghiệm feasible trong `T`; (ii) best official cost trong các run feasible. Không gán `0` hay bỏ im lặng những run không có nghiệm hợp lệ. Nếu official benchmark score chưa xác minh, đây là `AUTHOR_INPUT_NEEDED`.
- **Multi-objective endpoint:** hypervolume của feasible nondominated archive với normalization/reference point khóa từ development, hoặc một indicator khác được nhóm chọn trước; thêm objective vector/best trade-off, không chỉ một scalar. Chỉ số này phục vụ đánh giá NSGA-III, không dùng reference point học từ test.
- **Anytime và chi phí:** trace theo *thời gian thực* của best official score/indicator; generations, feasible rate, total call time, prompt/completion tokens, số review, event vs deadline triggers, no-op, invalid decision, invalid candidate, operator outcome. Giữ log cho từng `(instance, seed, method)`.
- **Đơn vị lặp độc lập:** một run trên một `(instance, seed)`. Dùng cùng danh sách seed cho cả ba phương pháp; instance là block, các generation/checkpoint trong cùng run **không phải n độc lập**. Pilot thời gian trên development; sau đó chốt số seed test theo ngân sách và báo độ biến thiên/CI. Không tự ghi sample size hay p-value trước khi chạy.
- **Hai contrast định trước:** E1 single–baseline, E2 multi–single. Báo hiệu ứng theo từng instance và tổng hợp có tính block; nếu kiểm định nhiều endpoint/instance, công bố cách kiểm soát multiplicity. Không chỉ so một biểu đồ trung bình hoặc tuyên bố thắng SOTA từ một tập instance nhỏ.
- **Tính tái lập:** khóa commit solver, phiên bản dataset/evaluator, prompt, schema, model/version, seed và config. Lưu raw decisions (ẩn dữ liệu nhạy cảm nếu có), reason codes của validator, tổng thời gian. Lỗi API/timeout là outcome có log, không xóa run sau khi thấy kết quả.

## 8. Những quyết định cần nhóm chốt trước khi triển khai

1. **Dataset/evaluator:** lấy các file ITC 2019 nào, có official cost/feasibility checker chính xác không, chia development/test thế nào?
2. **Objective vector:** NSGA-III tối ưu những mục tiêu nào, official score được tính và dùng làm primary endpoint/trigger ra sao?
3. **Vùng can thiệp:** biểu diễn một vùng trên một cá thể thế nào; thuật toán code tạo candidate regions/boundary và `K` candidate tối đa là gì? Nếu penalty không tách theo vùng thì fallback là gì?
4. **Operator library:** operator nào thực sự chạy được trên biểu diễn và constraint của dataset; execution budget được đếm theo evaluation/time/bước ra sao?
5. **Review policy:** `g0`, cửa sổ `W`, `Δmin`, `Δmax`, cooldown, default interval, stagnation/diversity metric/ngưỡng. Tất cả phải được khóa từ development.
6. **LLM contract:** model, temperature, token cap, prompts, schema JSON, timeout, validation/fallback. E2 Planner có thấy `StatePacket` gốc hay không? (Bản v2 đề xuất **có**.)
7. **Resource protocol:** wall-clock `T`, số seed, máy/chế độ API, cách xử lý rate limit và trường hợp không có feasible solution.

**Mức độ hoàn thiện:** đây là thiết kế đủ để phân chia module và làm pilot, **chưa phải** protocol có thể tái lập hoàn toàn khi bảy mục trên còn `AUTHOR_INPUT_NEEDED`. Bước kế tiếp duy nhất: nhóm chốt objective/evaluator và cách biểu diễn vùng trên một instance mẫu; hai quyết định này chi phối summary, trigger, operator và metrics.
