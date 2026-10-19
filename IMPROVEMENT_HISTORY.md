# Lịch sử cải thiện và quyết định

File này ghi quá trình thay đổi pipeline, không ghi ngày/giờ thực hiện và không giữ bảng số liệu chạy cũ. Mỗi mốc có mã H để truy vết; mã này là mục tài liệu, không phải Git commit. Trạng thái đang chạy và việc còn thiếu xem [PIPELINES.md](PIPELINES.md).

Khi cập nhật tiếp: thêm mốc mới gồm **vấn đề → thay đổi → quyết định/phạm vi → code/kiểm tra**. Nếu thay một quyết định cũ, ghi rõ mốc bị thay thế. Không sửa lịch sử thành như thể phiên bản mới đã luôn tồn tại.

R01–R07 nêu tại H21 đã được triển khai lần lượt trong H22–H28. Các giới hạn ở mốc cũ mô tả đúng phiên bản lúc đó; trạng thái hiện tại theo mốc mới hơn và `PIPELINES.md`.

Phần hiệu chỉnh còn mở R08 được đo lại tại H29–H30. Preset hiện có trạng thái riêng theo instance; không coi mọi cap đã là điểm hội tụ cuối cùng.

## H01. Chuyển baseline sang ITC 2019

- **Vấn đề:** baseline xuất phát từ dữ liệu/format riêng, chưa dùng trực tiếp XML ITC 2019.
- **Thay đổi:** thêm parser room/time/course/config/subpart/class/distribution/student; chromosome chọn option theo class, scorer và solution writer theo cấu trúc ITC.
- **Quyết định:** thiết kế trên instance đầy đủ; subset chọn các XML nguyên vẹn. Loại bỏ các module/data thuộc format cũ khỏi đường chạy hiện tại.
- **Truy vết:** [instance.py](NSGA-III/itc2019/instance.py), [scoring.py](NSGA-III/itc2019/scoring.py), [main.py](NSGA-III/main.py).

## H02. Tổ chức đầu vào và output theo cấu hình

- **Vấn đề:** chọn instance thủ công và thư mục timestamp làm kết quả khó tra, khó phân biệt cấu hình.
- **Thay đổi:** hỗ trợ `--instance`, `--manifest`, `--all`, `--inspect`; run ID dựa trên cấu hình, dữ liệu, code và môi trường; thêm config/summary/choices/trace.
- **Quyết định:** cùng định danh thì ghi đè sau khi run mới hoàn tất qua staging, không xóa kết quả đang có ngay lúc bắt đầu. Đổi seed/code/data/settings tạo run khác.
- **Truy vết:** [main.py](NSGA-III/main.py); [Guideline](NSGA-III/GUIDELINE.md) giải thích cách chạy và đọc output.

## H03. Dùng một solver chung cho ba pipeline

- **Vấn đề:** giữ baseline và các bản cải tiến/agent tách code có thể làm khác biệt đến từ solver thay vì controller.
- **Thay đổi:** hợp nhất parser, scorer, NSGA-III, repair và monitoring vào `NSGA-III/itc2019/`; agent folders chỉ giữ entry point và vai trò riêng.
- **Quyết định:** bỏ bản sao solver riêng; mọi thay đổi repair áp dụng cho cả baseline, single-agent và Inspector–Planner.
- **Truy vết:** ba `main.py`, hai `controller.py` của agent; [compare_pipelines.py](compare_pipelines.py) kiểm tra solver/config/hash/dependency phù hợp giữa các run.

## H04. Construction xét required distributions

- **Vấn đề:** greedy placement ưu tiên phòng/giờ có thể dựng một lịch khó sửa vì bỏ qua quan hệ bắt buộc giữa class.
- **Thay đổi:** lập danh sách required distributions theo class; khi mọi class liên quan đã đặt, đưa số vi phạm vào điểm chọn placement.
- **Quyết định:** giữ construction heuristic có giới hạn, không gọi nó là bộ giải CSP toàn cục. Forward checking trước khi đặt đủ nhóm vẫn chưa có.
- **Truy vết:** `_place` trong [nsga3.py](NSGA-III/itc2019/nsga3.py); phần chưa làm tương ứng R03 trong Pipeline.

## H05. Repair offspring tại chỗ

- **Vấn đề:** dựng lại toàn bộ timetable sau variation dễ phá những placement đang tốt và làm giảm tác dụng của crossover.
- **Thay đổi:** `RoomRepair` giữ chromosome offspring, dùng local hard repair để sửa quanh vi phạm.
- **Quyết định:** routine repair không trở thành một lần construction mới. Pymoo vẫn gọi routine này sau sampling đầu; stage timing ghi nhận cả lượt đó.
- **Truy vết:** `RoomRepair`, `PlacementSampling`, `_improve_hard` trong [nsga3.py](NSGA-III/itc2019/nsga3.py).

## H06. Joint moves thay cho chỉ single moves

- **Vấn đề:** một class không có nước đi giảm hard nếu class liên quan không đổi cùng.
- **Thay đổi:** thêm bounded two-class search, target từ violated distributions và room clashes; ưu tiên room domain có lợi cho `SameRoom`.
- **Quyết định:** chỉ commit joint move khi full score xác nhận giảm hard; không coi tìm hai class là đã giải được cả nhóm lớn.
- **Truy vết:** `_joint_improve_hard` trong [nsga3.py](NSGA-III/itc2019/nsga3.py); regression kiểm tra hard improvement, score accounting và deadline.

## H07. Nới acceptance trước khả thi, không làm hỏng nghiệm đã khả thi

- **Vấn đề:** yêu cầu mọi intervention phải lập tức đạt hard 0 loại bỏ cả bước sửa đúng hướng trên nghiệm chưa khả thi.
- **Thay đổi:** khi source hard > 0, nhận candidate giảm hard; khi source hard = 0, yêu cầu giữ khả thi và giảm soft total.
- **Quyết định:** kiểm tra trên bản sao và toàn instance; candidate hợp lệ vẫn phải qua NSGA-III survival. Quy tắc này dùng chung cho hai controller.
- **Truy vết:** `_admissible_candidate`, `apply_operator` trong [operators.py](NSGA-III/itc2019/operators.py); regression riêng cho acceptance.

## H08. Tách khả thi khỏi điều kiện dừng

- **Vấn đề:** vừa tìm được lịch khả thi không có nghĩa hết cơ hội tối ưu preferences.
- **Thay đổi:** dừng ở hard 0 thành cờ `--stop-on-feasible` tùy chọn; mặc định tiếp tục đến cap.
- **Quyết định:** không gắn yêu cầu xuất `solution.xml` với một điều kiện dừng bắt buộc. XML vẫn chỉ được ghi cho nghiệm cuối khả thi và qua kiểm tra sectioning nội bộ.
- **Truy vết:** `FeasibleSolutionTermination`, `run` trong [nsga3.py](NSGA-III/itc2019/nsga3.py), [main.py](NSGA-III/main.py); regression xác nhận default tiếp tục sau khả thi.

## H09. Giữ incumbent và cố định RNG của selection

- **Vấn đề:** nghiệm tốt đã thấy có thể biến mất khỏi quần thể cuối; tie-breaking không dùng đúng RNG có thể làm cùng seed cho đường chạy khác nhau.
- **Thay đổi:** monitor giữ best choices/quality, final selection đối chiếu incumbent với population; comparator dùng random state đã seed của solver.
- **Quyết định:** nghiệm đại diện chọn lexicographic `(hard, total)`, còn survival vẫn đa mục tiêu. Chưa có archive Pareto đầy đủ; việc ghi nhận candidate của LLM operator còn cần đồng bộ theo R01.
- **Truy vết:** `BaselineMonitor` trong [controller.py](NSGA-III/itc2019/controller.py), `_seeded_cv_tournament` và final selection trong [nsga3.py](NSGA-III/itc2019/nsga3.py).

## H10. Room-chain với neutral intermediate moves

- **Vấn đề:** room clash có thể cần một chuỗi đổi chỗ mà bước đầu không giảm hard.
- **Thay đổi:** thêm bounded beam/chain search qua trạng thái trung gian; xét cả room unavailable.
- **Quyết định:** không đưa trạng thái trung gian vào population; chỉ trả candidate cuối giảm tổng hard và lỗi phòng. Nhánh từng được cho budget lớn, sau được thu nhỏ và chuyển thành fallback ở H15/H16 vì full scoring tốn kém.
- **Truy vết:** `_room_chain_repair` trong [nsga3.py](NSGA-III/itc2019/nsga3.py).

## H11. Coordinated repair trong monitor

- **Vấn đề:** hard từ nhóm distribution có thể tồn tại dù mutation/routine repair đã chạy qua nhiều vòng.
- **Thay đổi:** tại mốc định kỳ, thử bounded coordinated rounds trên incumbent khi required-distribution violations chiếm ưu thế so với room clashes.
- **Quyết định:** dùng chung cả ba pipeline; không chạy thêm riêng cho agent. Giữ deadline và ghi riêng scoring calls/events; candidate cải thiện được ghi nhận trước survival.
- **Truy vết:** `_maybe_coordinated_repair` trong [controller.py](NSGA-III/itc2019/controller.py).

## H12. Room plateau search

- **Vấn đề:** room-heavy incumbent chưa gần khả thi nằm ngoài gate room-chain cũ.
- **Thay đổi:** thêm bounded search cho phép local neutral/worse moves, dùng room/required-distribution deltas trước khi full check; xét cả phòng không khả dụng.
- **Quyết định:** giữ hard slack, giới hạn số class thay đổi, deadline và số full checks; chỉ nhận cải thiện cuối. Nhánh hiện dùng ở vùng hard cao, không chạy trùng vai trò với room CP gần khả thi.
- **Truy vết:** [room_search.py](NSGA-III/itc2019/room_search.py), `_maybe_room_plateau_repair` trong [controller.py](NSGA-III/itc2019/controller.py).

## H13. Làm rõ scorer và chi phí đánh giá

- **Vấn đề:** student sectioning lặp phép kiểm tra xung đột giữa cặp class; timing tổng chưa cho biết stage nào tốn; cách đếm singleton `MaxBlock` cần sửa.
- **Thay đổi:** cache pair conflicts trong từng full score, không chia cache mù giữa các timetable; thêm stage timing/candidate counters; bỏ penalty block chỉ có một class trong nhánh `MaxBlock`.
- **Quyết định:** cache không thay decoder greedy/beam thành exact sectioning. Kiểm tra sectioning/output nội bộ không được gọi là xác nhận độc lập.
- **Truy vết:** [scoring.py](NSGA-III/itc2019/scoring.py), `PlacementProblem`/sampling/repair trong [nsga3.py](NSGA-III/itc2019/nsga3.py).

## H14. Progress và diversity có ý nghĩa cho review

- **Vấn đề:** nhiều chromosome khác nhau vẫn có thể gần giống nhau theo placement; soft total dao động khi chưa khả thi không phản ánh tiến bộ hard.
- **Thay đổi:** thêm placement diversity; progress theo hard trước khả thi và theo total sau khả thi; trigger một lần mỗi episode thay vì lặp liên tục.
- **Quyết định:** chromosome uniqueness và reference-direction coverage vẫn ghi log, nhưng không dùng làm low-diversity trigger hiện tại.
- **Truy vết:** `placement_diversity`, `population_metrics` trong [regions.py](NSGA-III/itc2019/regions.py), `_progress_signal`/`_trigger` trong [controller.py](NSGA-III/itc2019/controller.py).

## H15. Room-neighborhood CP-SAT mở sang blockers

- **Vấn đề:** chỉ thay class đang vi phạm có thể không đủ; class không vi phạm hiện tại đang chiếm các option thay thế cần thiết.
- **Thay đổi:** mở bounded neighborhood từ room conflicts/unavailability sang blocker classes; giữ ngoài vùng cố định, mã hóa room/pairwise required constraints và `MaxDays`/`MaxDayLoad`.
- **Quyết định:** CP là công cụ repair cục bộ, không thay NSGA-III và không chứng minh infeasibility toàn instance. Giữ incumbent option khi cắt domain; hạn chế table size/memory/time; full scorer kiểm tra constraints chưa mã hóa và student sectioning.
- **Truy vết:** [neighborhood.py](NSGA-III/itc2019/neighborhood.py), `_maybe_room_chain_repair` trong [controller.py](NSGA-III/itc2019/controller.py); regression cho blocker và global boundary.

## H16. Giới hạn chi phí và cooldown cho repair thất bại

- **Vấn đề:** gọi lại tìm kiếm đắt trên cùng cấu trúc lỗi nhưng không cải thiện làm tiêu ngân sách.
- **Thay đổi:** CP tối đa 32 class/32 option và 8 giây; fallback tối đa 2 giây/128 full checks; structural failure key và backoff 5 → 10 → 20 generations.
- **Quyết định:** không tăng vùng/timeout vô hạn khi probe thất bại; retry dựa trên cấu trúc lỗi/hard thay vì chỉ hash toàn chromosome. Cooldown chỉ hoãn retry, không loại nghiệm hay kết luận vô nghiệm.
- **Truy vết:** `RepairSettings`, `FailureBackoff` trong [repair_policy.py](NSGA-III/itc2019/repair_policy.py), logic gọi repair trong [controller.py](NSGA-III/itc2019/controller.py).

## H17. Tính đủ full search evaluations và thêm cap

- **Vấn đề:** chỉ đếm Pymoo candidate evaluations bỏ sót phần lớn công việc repair và kiểm tra controller.
- **Thay đổi:** cộng routine repair, monitor repair, region scoring, operator và rechecks vào tổng search count; thêm `--evaluations` và remaining-budget guards. Final verification được tách riêng.
- **Quyết định:** local delta/node CP không giả làm full evaluation; runtime vẫn tính chúng. Cap kiểm tra theo bước, chưa phải ngắt giữa mọi inner loop.
- **Truy vết:** `evaluation_count`/`FullEvaluationTermination` trong [nsga3.py](NSGA-III/itc2019/nsga3.py), [regions.py](NSGA-III/itc2019/regions.py), [controller.py](NSGA-III/itc2019/controller.py), [main.py](NSGA-III/main.py).

## H18. Chọn setting NSGA-III làm việc

- **Vấn đề:** tăng population/reference density hoặc repair intensity không mặc nhiên tốt hơn trong ngân sách hữu hạn.
- **Thay đổi được chọn:** từ population 40/H4/crossover 0.9 sang **20/H3/crossover 0.7**; giữ mutation rate 0.2, fraction 0.005, initial repair 12/3 và routine repair 8/2.
- **Phương án chưa chọn:** denser H5, inner reference layer H3+H2, tăng mutation, initial repair nhẹ/mạnh và routine repair 12/3. Không biến kết quả cục bộ của chúng thành mặc định chung.
- **Quyết định:** hai lớp directions vẫn là tùy chọn CLI; setting đã chọn không được gọi là tối ưu toàn ITC 2019. Mở CLI cho mutation/crossover/initial repair thay vì bắt sửa source.
- **Truy vết:** `Config`, `reference_directions` trong [nsga3.py](NSGA-III/itc2019/nsga3.py), parser trong [main.py](NSGA-III/main.py), [tune_solver.py](solver_checks/tune_solver.py).

## H19. Chọn review gate và action limits

- **Vấn đề:** checkpoint dày, event lặp và action budget nhỏ có thể làm review tốn chi phí nhưng ít cơ hội sửa.
- **Thay đổi được chọn:** interval mặc định 5 → 10; min interval 2 → 3; max interval 10 → 20; thêm diversity recovery threshold 0.30 bên cạnh trigger 0.20; giữ warm-up/window 5.
- **Action limits:** giữ tối đa 6 regions, tăng region size 3 → 5 và operator cap 8 → 24; review cap 12, timeout 15 giây/call. Giữ một thư viện operator chung, không tạo thêm quyền cho multi-agent.
- **Quyết định:** chọn qua offline replay và thử action library. Điều đó chưa kiểm tra được chất lượng chẩn đoán/lựa chọn của LLM thật.
- **Truy vết:** `ControllerSettings`/`_trigger`/`_packet` trong [controller.py](NSGA-III/itc2019/controller.py), [tune_review_gate.py](solver_checks/tune_review_gate.py), [tune_controller.py](solver_checks/tune_controller.py).

## H20. Profile ngân sách và chạy local thống nhất

- **Vấn đề:** một mốc thời gian chung không phù hợp cho mọi XML; lệnh thủ công dễ lệch setting/role-call budget giữa pipeline.
- **Thay đổi:** thêm `profiles/pilot.json`, `run_pilot.py`, budget theo instance; wrapper dùng cùng Python/solver/repair và cấp single/multi tối đa một/hai calls mỗi review.
- **Quyết định:** generation cap chỉ là một hàng rào, kết hợp evaluation/time cap. Việc hiệu chỉnh dùng trajectory progress và khoảng quan sát thêm; quy tắc dừng hiệu chỉnh trong script không phải điều kiện dừng tự động của baseline khi chạy bình thường.
- **Giới hạn:** budget local chưa xác nhận ổn định qua nhiều seed; ca bị time ceiling ngắt chưa chứng minh đạt plateau. Các con số cap đang dùng chỉ nằm trong JSON, không sao chép thành báo cáo benchmark.
- **Dependencies:** thêm OR-Tools; requirements của hai agent tham chiếu requirements baseline để không thiếu dependency repair chung.
- **Truy vết:** [run_pilot.py](run_pilot.py), [pilot.json](profiles/pilot.json), [select_pilot_budget.py](solver_checks/select_pilot_budget.py), ba requirements files.

## H21. Hợp nhất tài liệu, loại các báo cáo cũ

- **Vấn đề:** nhiều audit/calibration/comparison/progress documents chồng nhau, một số mô tả setting hoặc blocker đã thay đổi.
- **Thay đổi:** chỉ giữ hai tài liệu kỹ thuật chuẩn: `PIPELINES.md` nói code đang có và phần còn thiếu; file này nói quá trình thay đổi/quyết định. Gỡ các báo cáo số liệu/progress cũ và sửa liên kết từ README.
- **Quyết định:** không chép lại bảng kết quả cũ, không thêm ngày/giờ vào lịch sử. README và hướng dẫn thao tác vẫn giữ vai trò giới thiệu/cách chạy, không trở thành progress tracker khác. Profile chỉ giữ cấu hình chạy và provenance tối thiểu thay vì bảng số liệu hiệu chỉnh nhúng trong đó.
- **Phạm vi:** không thay logic solver, không chạy lại tuning/đánh giá hiệu năng và không xóa dataset, XML solutions, raw run artifacts hay các script kiểm tra. Các phần R01–R08 vẫn là việc chưa triển khai, không được ghi thành thành tích đã xong.

## H22. Một incumbent chung trước survival — R01

- **Vấn đề:** archive của monitor và controller/operator không thống nhất; candidate tốt có thể bị survival loại rồi biến mất khỏi progress và final selection.
- **Thay đổi:** `PlacementProblem.score_state` vừa chấm vừa ghi vào `Incumbent`, dùng chung cho population, repair, region và operator checks. Monitor đồng bộ trước/sau repair; controller dùng archive cho stagnation, recent outcomes và final output, tách rõ population metrics.
- **Quyết định:** archive một nghiệm đại diện lexicographic `(hard, total)`, không thay NSGA-III survival bằng single-objective selection và không gọi đây là Pareto archive. Source/candidate arrays được copy để tránh mutation làm hỏng archive.
- **Thay thế:** phần chưa đồng bộ ở H09/R01.
- **Truy vết/kiểm tra:** `Incumbent`, `score_state`, `run` trong [nsga3.py](NSGA-III/itc2019/nsga3.py); `_sync_incumbent`/outcome trong [controller.py](NSGA-III/itc2019/controller.py). Regression cố tình loại candidate cải thiện khỏi population nhưng vẫn giữ đúng archive.

## H23. Sửa required distributions theo nhóm — R02

- **Vấn đề:** single/two-class moves không vượt được một số bottleneck cần nhiều class đổi cùng; room-seeded CP không có target nếu chỉ còn distribution violations.
- **Thay đổi:** thêm `distribution_neighborhood_repair`, chọn nhóm required đang vi phạm làm seed cho finite-domain CP với fixed boundary. Monitor thử group CP trước; chỉ còn thời gian mới chạy pair fallback.
- **Quyết định:** nhánh này yêu cầu giảm tổng hard, không yêu cầu giảm room clash hay đạt ngay hard 0. Giữ deadline chung 5 giây và giới hạn full checks; không cấp thêm solver riêng cho agent.
- **Thay thế/mở rộng:** H06/H11 và R02; pair search vẫn giữ làm fallback rẻ.
- **Truy vết/kiểm tra:** [neighborhood.py](NSGA-III/itc2019/neighborhood.py), `_maybe_coordinated_repair` trong [controller.py](NSGA-III/itc2019/controller.py). Regression nhóm ba lớp SameAttendees không có room targets; diagnostic trực tiếp trên saved incumbent chỉ còn distribution violation.

## H24. Constraint-aware construction và forward checking — R03

- **Vấn đề:** construction cũ chờ đủ nhóm mới xét distribution; quyết định sớm có thể chặn các class chưa đặt.
- **Thay đổi:** `ConstructionIndex` dùng lại required-pair index/cache qua các cá thể; ưu tiên domain nhỏ/mức liên kết; kiểm tra pairs đã đặt và partial global costs. Forward checking tối đa bốn hàng xóm, lan truyền domains tương thích sau mỗi placement; giao domain SameRoom/SameStart khi có. Cắt mẫu theo cả time/room, không để một time nhiều room chiếm hết domain.
- **Quyết định:** tối đa 32 time/16 room lấy mẫu, giữ 64 placements/class; preserve diversity bằng seeded sampling. Hết deadline thì hoàn tất chromosome hợp lệ về option domain, không gọi phần còn lại là đã repair. Không biến sampled-domain failure thành infeasibility certificate.
- **Thay thế:** H04/R03. Không tăng population/H để che lỗi khởi tạo.
- **Truy vết/kiểm tra:** [construction.py](NSGA-III/itc2019/construction.py), sampling trong [nsga3.py](NSGA-III/itc2019/nsga3.py). Regression CommonStart; bounded construction/run probes trên sáu instance local có student data.

## H25. Adaptive room neighborhood và global block models — R04

- **Vấn đề:** trần vùng cố định bỏ sót blocker; MaxBreaks/MaxBlock trước đây chỉ bị phát hiện qua full check.
- **Thay đổi:** `adaptive_room_repair` tăng từ 16/16 lên preset 32/32 rồi tối đa 64/64 class/options trong một deadline/check budget. Bổ sung CP MaxBreaks/MaxBlock với outside fixed, gom day/week patterns, đếm cả class trùng interval và singleton đúng theo scorer.
- **Quyết định:** tổng room-CP deadline vẫn 8 giây; không nhân timeout theo số tầng. Table/model quá lớn trả resource/budget status; full scorer vẫn là điều kiện nhận cuối. Student sectioning không được nhúng vào placement CP.
- **Thay thế:** giới hạn cố định ở H15/H16/R04. Backoff và beam fallback vẫn giữ.
- **Truy vết/kiểm tra:** [neighborhood.py](NSGA-III/itc2019/neighborhood.py). Regression so chi phí CP với scorer trên mọi tổ hợp của fixture có fixed boundary, nhiều ngày/tuần, interval trùng; kiểm tra cả MaxDays/MaxDayLoad/MaxBreaks/MaxBlock. Diagnostic integrated room repair trên saved incumbent.

## H26. Cache scoring và guards bên trong repair — R05

- **Vấn đề:** recheck/region scoring lặp sectioning; repair quần thể đầu bị gọi hai lần; loop tốn kém khó dừng sát budget.
- **Thay đổi:** `ScoreEngine` cache 16 chromosome/instance, giữ score và assignments; mọi đường search scoring dùng cùng engine. Cache hits vẫn tính logical evaluation. Bổ sung physical/cache counters, inner deadline guards, kiểm tra remaining full checks trước joint/CP; chọn exhaustive theo product domain ≤ 96 thay vì số class toàn instance. Skip lượt repair Pymoo ngay sau sampling. `BudgetedNSGA3` thu nhỏ batch offspring cuối để không vượt evaluation cap.
- **Quyết định:** local delta không giả thành full score; không giảm logical charge nhờ cache. Reserve objective/recheck allowance trước repair/controller. Final verification độc lập vẫn tách riêng. Time cap vẫn hợp tác: một full score/initialization đã bắt đầu không bị cưỡng bức cắt giữa chừng.
- **Thay thế/mở rộng:** H05/H13/H17/R05. Không tuyên bố đã có incremental student scoring giữa hai chromosome khác nhau.
- **Truy vết/kiểm tra:** [scoring.py](NSGA-III/itc2019/scoring.py), [nsga3.py](NSGA-III/itc2019/nsga3.py), [main.py](NSGA-III/main.py), các repair helpers. Regression cache bound/hit charging/expired deadline, cap không chia hết population; đối chiếu logical requests với `evaluation_count` cho baseline và hai controller giả, cùng các real-instance probes.

## H27. WHERE dựa trên quan hệ thật, HOW sửa đồng thời cả vùng — R06

- **Vấn đề:** hai representative theo total ít thể hiện trade-offs; attribution distribution chia thô; destroy/repair tuần tự ba lớp không vượt được joint barrier.
- **Thay đổi:** representative thứ hai ưu tiên objective/placement distance khi hard gần tương đương; grow vùng theo connected frontier. Pair violations quy đúng endpoints; student-conflict edges lấy từ decoded assignments. Boundary giữ cả quan hệ đang thỏa và possible room blockers. `destroy_repair` thử CP trên toàn vùng agent đã chọn, fixed outside, rồi fallback tuần tự nếu còn budget.
- **Quyết định:** không mở rộng target agent ra ngoài vùng đã xác minh. CP tối đa 3 giây, dùng cùng evaluation allowance; full scoring/acceptance dùng chung. Public boundary list cắt ở 32 IDs nhưng có total/truncated, nội bộ vẫn giữ đầy đủ. Global-group attribution vẫn heuristic; soft CP objective ưu tiên time/room, full scorer kiểm các soft thành phần còn lại.
- **Thay thế:** mô tả WHERE/HOW cũ tại R06/H19; giữ cùng action space cho single/multi, baseline tiếp tục có group-repair primitive chung.
- **Truy vết/kiểm tra:** [regions.py](NSGA-III/itc2019/regions.py), [operators.py](NSGA-III/itc2019/operators.py). Regression bốn lớp cần đổi đồng thời mới giảm soft mà vẫn khả thi, giữ nguyên class ngoài vùng, survival reject không làm mất archive; attribution không đổ lỗi lên endpoint không vi phạm.

## H28. Capacity-plan sectioning fallback — R07

- **Vấn đề:** greedy có thể dùng hết section cần cho người sau; beam 12 có thể loại parent choice duy nhất hợp lệ. Chỉ đổi placement không sửa được lỗi này.
- **Thay đổi:** nếu greedy còn unassigned, sinh bundles theo parent-topological subparts và giải CP số lượng người dùng từng bundle; gán lại toàn bộ người đăng ký course bị lỗi. Cache structural capacity plan theo course, ghép lại với student histories của timetable hiện tại và tính lại conflict/missing counts.
- **Quyết định:** bounded 4.096 bundles/200.000 nodes; CP một worker, seed 0, deterministic limit 0,5 để scoring lặp lại được. Plan chỉ dùng class limits và structural choices, không phụ thuộc lịch nên có thể cache. Không có plan trong tập đã sinh không chứng minh instance vô nghiệm; không bảo đảm tối ưu student penalty.
- **Thay thế:** thiếu reassignment ở H13/R07. Giữ greedy nhanh khi đã gán đủ; chỉ bật fallback khi cần.
- **Truy vết/kiểm tra:** [sectioning.py](NSGA-III/itc2019/sectioning.py), [scoring.py](NSGA-III/itc2019/scoring.py), per-instance cache trong [instance.py](NSGA-III/itc2019/instance.py). Regression release capacity bằng reassignment, recover root ngoài beam, deterministic repeat và ca thiếu capacity thật vẫn phải báo unassigned.
- **Đồng bộ:** thêm hai module mới vào source fingerprint/output metadata; sửa Pipeline/README/Guideline. Giữ preset caps cũ để chạy local nhưng đánh dấu provisional/recalibration-required, không gắn hash solver mới với phép hiệu chỉnh cũ. Không dựng lại báo cáo số liệu đã gỡ ở H21.

## H29. Chọn lại cấu hình trên solver đã tăng cường — R08

- **Vấn đề:** setting/budget của H18–H20 được chọn trước thay đổi construction, group CP, scoring cache và sectioning; không thể xem như đã hiệu chỉnh cho source mới.
- **Phép đo:** giữ nguyên core solver; sàng lọc sáu cấu hình trên hai instance/hai seed ở cùng evaluation cap, sau đó thử hai ứng viên trên ba ca khó với seed mới 73. Thử thêm tổ hợp population nhỏ + mutation, ba instance dễ hơn, các CP budgets và hai seed xác nhận cho toàn bộ sáu instance. Các lượt so đều ghi đầy đủ config, XML/source hashes, dependency versions, CPU/wall time, trajectory và repair events.
- **Quyết định:** preset chọn mutation rate 0,35/class fraction 0,01. `nbi-spr18` và `yach-fal17` dùng population 12/H=2; các file khác giữ 20/H=3. Giữ crossover 0,7, initial repair 12/3, offspring repair 8/2 và không bật inner layer. Đây là lựa chọn trong miền cấu hình đã thử, không tuyên bố tối ưu toàn ITC 2019.
- **Repair budget:** giữ room-CP 8 giây/deterministic limit 0,5 cho phần lớn instance. Riêng `yach` chọn 15 giây/2 sau xác nhận hai seed: không giảm chất lượng hard ở cặp đã đo và có cải thiện ở một seed. Thử tăng thời gian trên trajectory khác không tự động tốt hơn; snapshot cố định cho thấy limit thấp có thể cắt trước một nước sửa hữu ích. Không áp một kết quả snapshot cho mọi instance.
- **Phương án không chọn:** hai lớp directions, tăng repair intensity, mở vùng rộng hơn và gọi room CP sớm ở ngưỡng hard 40 không cho lợi ích đủ ổn định trong phạm vi thử. Đặc biệt ngưỡng 40 thay đổi nhánh nào được chạy, không chỉ bổ sung thời gian; không giữ nó chỉ vì nghe có vẻ mạnh hơn.
- **Thực thi:** preset nằm trong `profiles/pilot.json`, còn CLI defaults của `main.py`/`Config` giữ nguyên. `run_pilot.py` hỗ trợ solver và `repair_*` overrides theo instance, phát mỗi flag một lần và áp như nhau cho baseline/single/multi. Không tạo solver yếu hơn riêng cho baseline.
- **Kiểm tra:** bổ sung regression cho override, horizon/censoring và setting consistency theo seed. Warm native CP runtime trước các unit test có deadline ngắn; calibration vẫn tính cold-start thực của từng worker. Không gọi API, không chỉnh thuật toán, không sửa dữ liệu.
- **Truy vết:** [recalibrate_solver.py](solver_checks/recalibrate_solver.py), [analyze_recalibration.py](solver_checks/analyze_recalibration.py), [run_pilot.py](run_pilot.py), `solver_checks/recalibration_v4/`. Manifest giữ config đã đóng băng để replay sau khi live profile đổi; guard từ chối source/environment khác. File `_accepted_records.json` chỉ rõ records dùng chọn cap, bao gồm cặp `yach` có CP allowance mới.

## H30. Chọn ngân sách đo được và phân loại nút thắt còn lại

- **Vấn đề:** một smoke run hoặc một seed không phân biệt được thiếu budget với repair bị kẹt. Cũng không được gọi endpoint chạm time cap là đã hội tụ.
- **Phạm vi đo:** xác nhận sáu instance ở seed 17/42, giữ tiếp tìm kiếm sau hard 0. Các instance thông thường được đo tới 1.200 evaluations; `bet` có trần thử 2.400. Một lượt `bet` chạm time cap trước trần evaluations; ghi nguyên censoring, không ngoại suy thành một trajectory đã đo đủ.
- **Quy tắc evaluation cap:** sau khả thi, xét mốc đạt 90% soft gain quan sát được, cộng khoảng đệm và ít nhất 400 evaluations sau khả thi; khi chưa khả thi, xét lần giảm hard gần nhất và khoảng đệm 400. Làm tròn theo 100, không vượt quãng đã quan sát ở cả hai seed. Đây là cap làm việc theo dữ liệu hữu hạn, không phải bằng chứng chạy thêm vô ích.
- **Quy tắc time cap:** lấy thời gian tới cap trên seed chậm hơn, nhân 1,35, cộng dự phòng hai generation dài và tối đa 360 giây API latency cho Inspector–Planner. Dùng cùng cap cho ba pipeline của cùng instance. Đây là safety ceiling; thời gian từ các job kỹ thuật chạy đồng thời không phải phép so tốc độ solver và không phải đo API thật.
- **Trạng thái còn mở:** giữ `progressing_at_evaluation_cap` cho `muni-fi` và `yach`; giữ `time_censored` cho `bet`. `recalibration_required` vẫn true, không gỡ cờ để gọi toàn bộ preset là đã hội tụ. `mary`, `muni-fsps`, `nbi` có local knee được quan sát trong phạm vi đo. Chi tiết caps/status chỉ giữ trong JSON để tránh nhiều bảng số liệu lệch nhau.
- **Chẩn đoán hard:** bốn instance đạt khả thi nội bộ ở cả hai seed xác nhận. `yach` còn room clashes; `bet` còn phòng và required-distribution violations cùng tồn tại. Các endpoint này không bị thiếu student assignments. Ở `bet`, cả gate gần khả thi của room CP và gate distribution-dominant đều có thể không mở, nên plateau/routine repair giữ vai trò chính.
- **Probe sau run:** kiểm tra riêng trên choices cố định, không trộn điểm của probe vào điểm GA. Một bước CP trên `yach` giảm được hard; bước tiếp theo vẫn chưa tìm được cải thiện trong budget. Ép room/group neighborhoods trên incumbent `bet` cũng chưa tìm được candidate ở giới hạn đã thử. `UNKNOWN` là chưa tìm được nghiệm/chứng minh trong model/budget, không phải instance vô nghiệm; `INFEASIBLE` của vùng cắt domain cũng không chứng minh toàn timetable vô nghiệm.
- **Việc còn lại:** sửa hiệu quả/bao phủ neighborhood, model và phối hợp repair cho xung đột phòng/required constraints của hai ca khó. Không suy ra chỉ tăng H hoặc generations sẽ khắc phục. Validator độc lập và LLM thật tiếp tục để riêng, chưa thực hiện trong đợt này.
- **Truy vết:** `_budget_suggestion.json`, `_accepted_records.json`, `_pilot_before_recalibration.json` và các JSON/NPZ dưới `solver_checks/recalibration_v4/`; các probe dùng [probe_neighborhood.py](solver_checks/probe_neighborhood.py), [check_strengthening.py](solver_checks/check_strengthening.py). Chỉ hai tài liệu Pipeline/Lịch sử giữ mô tả kỹ thuật/quyết định, không dựng lại báo cáo Markdown số liệu.

## H31. Gemini Free Tier thay adapter OpenAI

- **Vấn đề:** nhóm không có ngân sách OpenAI; adapter Responses API cũ không đổi sang Gemini chỉ bằng thay key/model. Lỗi hạ tầng không được diễn giải như quyết định no-op của model.
- **Thay đổi:** `GeminiDecisionClient` dùng Google GenAI SDK/native structured outputs, giữ chung PROMPTS/SCHEMAS và giao diện trả decision; chỉ nhận model trong Free Tier allowlist, không dùng latest/preview aliases hay fallback provider. Cả single và Inspector-Planner dùng cùng transport/policy/model.
- **An toàn chi phí:** key lấy từ environment, bắt xác nhận project Free Tier, không bật billing. Allowlist không chứng minh project miễn phí; code không thể suy billing tier từ key. SDK một attempt, request pacing 10 giây, input cap 60.000 ký tự, timeout tối đa 15 giây trong preset. Output allowance đổi 300 → 1.024, low thinking để chứa bounded JSON + thinking; chưa tuning bằng model thật. Giữ nguyên solver/repair parameters H29.
- **Theo dõi lỗi:** log provider, requested/returned model, finish reason, prompt/output/thinking/total/cached tokens và số requests thực. Tắt automatic continuation/function calling của SDK để không sinh request ẩn ngoài budget; HTTP 429 không được SDK tự retry. Invalid JSON/schema không chạy operator. Lỗi quota/billing/auth/network/timeout dừng run, lưu incomplete và exit code 3; wrapper ngừng lượt kế tiếp, comparator từ chối diễn giải run lỗi hoặc không gọi API.
- **Truy vết/kiểm tra:** llm.py, main.py, controller.py, run_pilot.py, compare_pipelines.py, profiles/pilot.json; requirements đổi sang `google-genai`. Tests offline/mock HTTP xác nhận native serialization, schema/type guards, deadline/pacing, usage, quota/no-retry và so sánh khi hard > 0. Thêm preflight offline, chỉ `--live` mới gọi một packet tổng hợp.
- **Giới hạn:** người dùng sẽ đặt key sau; chưa gửi request Gemini thật, chưa đo chất lượng/latency model. Profile giữ hash snapshot calibration cũ cùng chú thích thay đổi source; không dùng hash adapter mới làm bằng chứng cho phép đo cũ.

## H32. Chuẩn bị đối chiếu validator chính thức

- **Vấn đề:** `hard=0`/score do scorer nội bộ không chứng minh đúng semantics/penalty ITC. Validation và bước công bố nghiệm trên website là hai thao tác khác nhau.
- **Thay đổi:** thêm `tools/validate_itc2019.py`: re-score/export saved feasible choices, submit có authentication tới đúng validation endpoint, hoặc import raw JSON response do người dùng lưu. Future feasible run xuất `expected.json` cạnh XML; receipt lưu riêng, gắn hash XML và đối chiếu instance/assigned class count/total.
- **Quyết định:** không chạy lại GA để chuẩn bị file, không tự công bố solution. `OK` nhưng khác điểm là `penalty_mismatch`; response lạ/custom/incomplete không thành pass. Comparator từ chối receipt sai hash/score; `not_run` được giữ khi chưa có xác nhận.
- **Chuẩn bị thực tế:** đã chấm lại/export hai saved incumbents khả thi của `muni-fsps-spr17` và `nbi-spr18`, giữ nguyên score đã lưu. Website yêu cầu đăng nhập; người dùng có tài khoản và sẽ Validate sau. Vì vậy chưa có official pass và không gửi token/credentials qua chat.
- **Truy vết/kiểm tra:** tools/validate_itc2019.py, main.py, compare_pipelines.py, test_official_validation.py; artifacts dưới `solver_checks/official_validation/` được gitignore. Tests kiểm response status, score mismatch, XML hash binding, malformed payload và không network khi thiếu authentication; không gọi chúng là tests thay thế official scorer.
