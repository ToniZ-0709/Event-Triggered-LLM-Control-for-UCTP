"""Build two Vietnamese technical reports without modifying solver or Markdown.

Sources: the canonical pipeline/history Markdown and the measured JSON profile.
The authored report text is an editorial condensation, not a new experiment.
"""
from pathlib import Path
import hashlib
import html
import json
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether,
    Table, TableStyle, Image,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf"
ASSETS = OUT / "assets"
QA = ROOT / "tmp/pdfs"
PROFILE = json.loads((ROOT / "profiles/pilot.json").read_text(encoding="utf-8"))
BLUE, TEAL, INK = "#254F73", "#137F83", "#203448"
GRAY, LIGHT, PALE = "#586B7A", "#F0F5F9", "#EAF6F4"
PAGE_W, PAGE_H = A4
MARGIN = 43
WIDTH = PAGE_W-2*MARGIN

for name, file in [("Arial","arial.ttf"),("Arial-Bold","arialbd.ttf"),
                   ("Arial-Italic","ariali.ttf"),("Arial-BoldItalic","arialbi.ttf")]:
    pdfmetrics.registerFont(TTFont(name,str(Path("C:/Windows/Fonts")/file)))
pdfmetrics.registerFontFamily("Arial",normal="Arial",bold="Arial-Bold",
                            italic="Arial-Italic",boldItalic="Arial-BoldItalic")

ST = {
    "body":ParagraphStyle("body",fontName="Arial",fontSize=10.1,leading=14.5,
                          textColor=colors.HexColor(INK),spaceAfter=6),
    "bullet":ParagraphStyle("bullet",fontName="Arial",fontSize=10,leading=14,
        textColor=colors.HexColor(INK),leftIndent=11,firstLineIndent=-10,spaceAfter=4),
    "small":ParagraphStyle("small",fontName="Arial",fontSize=8.5,leading=11.8,
                           textColor=colors.HexColor(GRAY),spaceAfter=5),
    "caption":ParagraphStyle("caption",fontName="Arial-Italic",fontSize=8.4,
        leading=11.4,textColor=colors.HexColor(GRAY),spaceAfter=10),
    "title":ParagraphStyle("title",fontName="Arial-Bold",fontSize=25,leading=29,
        textColor=colors.HexColor(BLUE),spaceAfter=12),
    "section":ParagraphStyle("section",fontName="Arial-Bold",fontSize=18,leading=23,
        textColor=colors.HexColor(BLUE),spaceAfter=11),
    "h2":ParagraphStyle("h2",fontName="Arial-Bold",fontSize=11.3,leading=15,
        textColor=colors.HexColor(TEAL),spaceBefore=9,spaceAfter=5,keepWithNext=True),
    "eyebrow":ParagraphStyle("eyebrow",fontName="Arial-Bold",fontSize=8.3,leading=11,
        textColor=colors.HexColor(TEAL),spaceAfter=8),
    "cell":ParagraphStyle("cell",fontName="Arial",fontSize=8.8,leading=12,
        textColor=colors.HexColor(INK)),
    "headcell":ParagraphStyle("headcell",fontName="Arial-Bold",fontSize=8.6,
        leading=11.8,textColor=colors.white),
    "code":ParagraphStyle("code",fontName="Courier",fontSize=8.4,leading=12,
        textColor=colors.HexColor(INK),spaceAfter=3),
    "history":ParagraphStyle("history",fontName="Arial",fontSize=9.6,leading=13.4,
        textColor=colors.HexColor(INK),leftIndent=11,firstLineIndent=-10,spaceAfter=3),
}


def clean(s):
    return s.replace("—","-").replace("–","-").replace("\u2011","-")


def markup(s):
    s=html.escape(clean(s))
    s=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",s)
    s=re.sub(r"`([^`]+)`",r'<font name="Arial-Bold">\1</font>',s)
    return s


def p(s,style="body"):
    return Paragraph(markup(s),ST[style])


def bullets(items,style="bullet"):
    return [p("• "+s,style) for s in items]


def h(s):
    return p(s,"h2")


def title(index,title,sub=None,cover=False):
    seq=[p(index,"eyebrow"),p(title,"title" if cover else "section")]
    if sub:
        seq.append(p(sub,"small"))
    return seq


def fig(name,height,caption):
    path=ASSETS/(name+".png")
    im=Image(str(path))
    im.drawWidth=WIDTH
    im.drawHeight=height
    return [im,p(caption,"caption")]


def callout(heading,body):
    t=Table([[p(heading,"h2")],[p(body)]],colWidths=[WIDTH-20])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(PALE)),
        ("BOX",(0,0),(-1,-1),.5,colors.HexColor("#C9E1DE")),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,-1),(-1,-1),9),
    ]))
    return [t,Spacer(1,7)]


def table(headers,rows,widths):
    data=[[p(c,"headcell") for c in headers]]+[[p(str(c),"cell") for c in row] for row in rows]
    t=Table(data,colWidths=[WIDTH*w for w in widths],repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor(BLUE)),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor(LIGHT)]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor(BLUE)),
        ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#DFE7EE")),
    ]))
    return [t,Spacer(1,7)]


class NumberedCanvas(canvas.Canvas):
    def __init__(self,*args,report_name="",**kwargs):
        super().__init__(*args,**kwargs)
        self.saved=[]
        self.report_name=report_name

    def showPage(self):
        self.saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        count=len(self.saved)
        for state in self.saved:
            self.__dict__.update(state)
            self.setStrokeColor(colors.HexColor("#DCE5EC"))
            self.setLineWidth(.6)
            self.line(MARGIN, PAGE_H-30, PAGE_W-MARGIN, PAGE_H-30)
            self.line(MARGIN, 32, PAGE_W-MARGIN,32)
            self.setFont("Arial-Bold",8)
            self.setFillColor(colors.HexColor(BLUE))
            self.drawString(MARGIN,PAGE_H-22,"NSGA-III  /  ITC 2019")
            self.setFont("Arial",8)
            self.setFillColor(colors.HexColor(GRAY))
            self.drawRightString(PAGE_W-MARGIN,PAGE_H-22,self.report_name)
            self.drawString(MARGIN,20,"Báo cáo kỹ thuật nội bộ  |  NSGA-III với repair dùng chung")
            self.drawRightString(PAGE_W-MARGIN,20,f"{self._pageNumber} / {count}")
            super().showPage()
        super().save()


def build(name,pages,report_name):
    story=[]
    for page in pages:
        if story:
            story.append(PageBreak())
        story.extend(page)
    target=OUT/name
    doc=SimpleDocTemplate(str(target),pagesize=A4,
        leftMargin=MARGIN,rightMargin=MARGIN,topMargin=47,bottomMargin=45,
        title=report_name,author="Nhóm phát triển solver",subject="NSGA-III / ITC 2019")
    doc.build(story,canvasmaker=lambda *a,**kw:NumberedCanvas(*a,report_name=report_name,**kw))
    return target


def pipeline_pages():
    pages=[]
    page=title("BÁO CÁO 01  /  KIẾN TRÚC VÀ VẬN HÀNH","Solver hiện tại","Mô tả implementation đang có, không phải thiết kế dự kiến hay báo cáo benchmark.",True)
    page+=callout("Kết luận chính","Ba pipeline dùng cùng một NSGA-III có repair. LLM chỉ bổ sung quyết định WHERE, HOW và WHEN; không thay scorer, representation hay cơ chế chọn lọc của solver.")
    page+=fig("solver_overview",300,"Hình 1. Luồng rút gọn của solver. Full scoring cập nhật incumbent trước survival; cap bao gồm generation, logical evaluations và thời gian. Controller được bỏ qua ở baseline.")
    page += bullets([
        "**Baseline:** NSGA-III + construction/repair/monitor chung; không gọi API.",
        "**Single-agent:** một agent chọn vùng, operator/budget và lần review tiếp theo.",
        "**Inspector-Planner:** Inspector chọn WHERE; Planner chọn HOW và WHEN sau khi target được xác minh.",
        "**Phạm vi báo cáo:** dữ liệu và scorer; search/repair; controller; ngân sách; output; trạng thái và giới hạn.",
    ])
    page += [p("Nguồn chính: PIPELINES.md; IMPROVEMENT_HISTORY.md (H01-H30); profiles/pilot.json. Giữ nguyên các file nguồn và logic solver trong lần biên tập này.","small")]
    pages.append(page)

    page=title("01  /  DỮ LIỆU VÀ ĐÁNH GIÁ","Một chromosome biểu diễn điều gì?")
    page += bullets([
        "**Đơn vị dữ liệu:** mỗi XML ITC 2019 là một instance độc lập. Pilot/subset chọn các XML nguyên vẹn, không cắt tùy ý class hoặc ràng buộc trong một instance.",
        "**Chromosome dài 2n:** n chỉ số time-option rồi n chỉ số room-option. Class không cần phòng dùng room index -1. Gene chọn trong domain cho sẵn của từng class.",
        "**Student sectioning không nằm trong gene:** scorer giải mã việc phân sinh viên theo course/config/subpart, parent relationships và class limits.",
    ])
    page += [h("Hard và soft được tách riêng")]
    page += table(["Thành phần","Ý nghĩa"],[
        ["room_conflicts","Các cặp class trùng phòng và trùng lịch."],
        ["room_unavailable","Placement giao với khoảng phòng không khả dụng."],
        ["hard_distributions","Vi phạm các distribution constraints bắt buộc."],
        ["unassigned_requests","Yêu cầu đăng ký course chưa được gán đủ."],
    ],[.31,.69])
    page += bullets([
        "**Hard** bằng tổng bốn thành phần trên. Hard 0 chỉ là khả thi theo implementation nội bộ.",
        "**Bốn objective:** weighted time, room, soft distribution và student conflict. NSGA-III survival vẫn đa mục tiêu; total penalty chỉ chọn nghiệm đại diện và đo tiến bộ sau khả thi.",
    ])
    page += [h("Sectioning: greedy trước, capacity-plan fallback khi cần")]
    page += bullets([
        "Greedy xử lý sinh viên nhiều course trước; beam 12 chọn bundle class. Nếu còn missing requests, giải lại toàn bộ người đăng ký course bị lỗi.",
        "Sinh bundle hợp lệ → CP-SAT tìm số người dùng mỗi bundle → ghép lại sinh viên theo xung đột với course khác → tính lại missing/conflicts.",
        "Giới hạn 4.096 bundles / 200.000 nodes; CP một worker, seed cố định, deterministic budget 0,5. Structural capacity plan được cache theo instance/course.",
        "Không tìm được plan trong tập bundle đã sinh không chứng minh XML vô nghiệm. Fallback ưu tiên sức chứa, chưa bảo đảm student penalty tối ưu.",
    ])
    page += [p("Truy vết: instance.py; scoring.py; sectioning.py. Bản đồ module chi tiết ở cuối báo cáo.","small")]
    pages.append(page)

    page=title("02  /  VÒNG LẶP TIẾN HÓA","Từ construction đến nghiệm đại diện")
    page += [h("1. Construction có xét quan hệ ràng buộc")]
    page += bullets([
        "Ưu tiên domain nhỏ, mức liên kết và seeded tie-break; xét required pairs với class đã đặt, partial global costs, room usage và unavailable.",
        "Forward checking tối đa 4 hàng xóm; lan truyền pair-compatible domains. SameRoom/SameStart tận dụng giao domain nếu có.",
        "Mỗi class lấy mẫu tối đa 32 time / 16 room, giữ tối đa 64 placements đa dạng. Domain bị cắt chỉ là giới hạn tìm kiếm, không phải bằng chứng vô nghiệm.",
    ])
    page += [h("2. Initial repair và variation")]
    page += bullets([
        "Initial repair dùng single moves rồi bounded two-class moves. Bỏ lượt RoomRepair tự động ngay sau sampling để tránh repair hai lần.",
        "Crossover giữ cặp time/room theo class; mutation thay một phần nhỏ class. Offspring repair sửa tại chỗ quanh vi phạm, không dựng lại toàn timetable.",
        "Routine joint search xét exhaustive khi product domain ≤ 96; domain lớn dùng lấy mẫu. Deadline được kiểm bên trong vòng lặp.",
    ])
    page += [h("3. Full scoring và NSGA-III survival")]
    page += bullets([
        "Full scorer đánh giá placement, required/soft distributions và decoded student assignment. NSGA-III dùng feasibility handling và bốn objective để chọn quần thể tiếp.",
        "Tournament/selection dùng RNG của solver; seed được ghi cùng config. Không thay survival bằng xếp hạng một scalar total.",
    ])
    page += [h("4. Incumbent dùng chung trước survival")]
    page += bullets([
        "Mọi candidate đi qua PlacementProblem.score_state đều có thể cập nhật incumbent theo thứ tự (hard, total), kể cả candidate do repair hoặc operator sinh ra.",
        "Candidate tốt bị survival loại vẫn còn trong incumbent. Monitor, controller feedback, progress và output cùng tham chiếu archive này.",
        "Archive hiện giữ một nghiệm đại diện; chưa có external Pareto archive đầy đủ.",
    ])
    page += callout("Hard 0 không phải điều kiện dừng bắt buộc","Mặc định solver tiếp tục cải thiện soft sau khi khả thi. Chỉ bật dừng sớm bằng --stop-on-feasible; nếu không, dừng khi chạm một trong các cap đã cấu hình.")
    pages.append(page)

    page=title("03  /  REPAIR DÙNG CHUNG","Các nhánh sửa được gọi như thế nào?")
    page+=fig("repair_routes",270,"Hình 2. Các gate chính trong monitor, theo thứ tự xét. Sơ đồ rút gọn không liệt kê mọi guard: remaining budget, duplicate source, deadline và cooldown vẫn được kiểm tra.")
    page += table(["Nhánh","Budget và acceptance"],[
        ["Distribution group CP","Mốc 5 generations; deadline chung 5 giây. Còn thời gian mới thử tối đa 3 rounds / 8 pairs mỗi round. Chỉ cần giảm tổng hard."],
        ["Room plateau","Tối đa 5 giây / 16 full checks; cho phép neutral/worse intermediate states. Chỉ trả cải thiện cuối."],
        ["Adaptive room CP","16/16 → preset 32/32 → tối đa 64/64 class/options trong cùng deadline/check budget. Mặc định 8 giây; yach dùng 15 giây theo profile."],
        ["Beam fallback / backoff","Fallback tối đa 2 giây / 128 full checks nếu còn budget. Cấu trúc lỗi thử thất bại được hoãn 5 → 10 → 20 generations."],
    ],[.28,.72])
    page+=bullets([
        "Room neighborhood mở sang lớp đang chặn phương án thay thế, kể cả lớp hiện không vi phạm; phần ngoài vùng được cố định.",
        "CP mã hóa room clash/unavailability, pairwise required constraints, MaxDays, MaxDayLoad, MaxBreaks và MaxBlock, kể cả fixed-boundary interactions.",
        "Room branch phải giảm tổng hard và lỗi phòng. Full scorer kiểm lại toàn timetable/sectioning; CP không thay thế scorer hay giải toàn instance.",
        "UNKNOWN là chưa giải xong trong model/budget. INFEASIBLE của vùng đã cắt domain cũng không chứng minh toàn XML vô nghiệm.",
    ])
    pages.append(page)

    page=title("04  /  WHERE VÀ WHEN","Từ search state đến vùng can thiệp")
    page+=fig("region_boundary",200,"Hình 3. Minh họa cấu trúc region/boundary, không phải timetable lấy từ dataset. Các node trong vùng được phép đổi; quan hệ với phần ngoài vẫn phải được bảo toàn qua validation và full scoring.")
    page += [h("WHERE: StatePacket được solver dựng, không do LLM tự đọc toàn lịch")]
    page += bullets([
        "Chọn tối đa hai representatives: nghiệm tốt nhất theo (hard,total) và một chromosome khác, ưu tiên trade-off objective/placement khi hard gần tương đương.",
        "Tạo tối đa 6 connected regions, mặc định 5 class/vùng. Pairwise violations quy về đúng endpoints; student conflicts lấy từ decoded assignment.",
        "Global-group attribution vẫn heuristic; missing course requests đánh dấu unlocalized, không gán cho một vùng tùy ý.",
        "Boundary chứa distribution relationships và alternative-room blockers. Packet hiển thị tối đa 32 boundary IDs kèm count/truncated; nội bộ giữ đầy đủ.",
        "Packet gồm progress/diversity, regions và allowed moves, 5 outcomes gần nhất, action limits. Incumbent metrics và population metrics được phân biệt.",
    ])
    page += [h("WHEN: trigger theo sự kiện và checkpoint")]
    page += bullets([
        "Warm-up generation 5; khoảng cách tối thiểu 3 generations. Checkpoint ban đầu 10; agent đặt interval tiếp theo trong [3,20].",
        "Review khi đến checkpoint, stagnation 5 generations hoặc placement diversity ≤ 0,20. Không gọi ở mọi generation.",
        "Low-diversity chỉ rearm khi diversity > 0,30; stagnation rearm sau tiến bộ. Progress theo hard trước khả thi, total sau khả thi.",
        "Tối đa 12 reviews; profile cấp single 12 calls, multi 24 calls; timeout 15 giây/call, output 300 tokens. Baseline không gọi API.",
    ])
    pages.append(page)

    page=title("05  /  HOW VÀ VALIDATION","Agent đề xuất; solver quyết định thực thi")
    page+=fig("controller_roles",232,"Hình 4. Single-agent và Inspector-Planner dùng cùng StatePacket, action space và safety rules. Ở multi-agent, target được xác minh trước Planner; thời gian và tokens của cả hai calls đều được ghi.")
    page += table(["Action","Thực thi trong vùng được xác minh"],[
        ["move","Đổi time hoặc room của một class."],
        ["swap","Hoán đổi hai placements tương thích với domain của cả hai class."],
        ["destroy_repair","CP sửa đồng thời toàn vùng, tối đa 3 giây; còn evaluation budget mới fallback tuần tự tối đa 3 class."],
        ["no_op","Không đổi timetable; vẫn có next review interval hợp lệ."],
    ],[.25,.75])
    page += bullets([
        "**Kiểm tra response:** action, IDs, target/boundary, operator, evaluation budget và next-review interval đều phải hợp lệ. Agent không chạy mã tự sinh.",
        "**Source chưa khả thi:** candidate phải giảm hard, không bắt lập tức về hard 0. **Source khả thi:** giữ hard 0 và giảm soft total.",
        "Áp operator trên bản sao, kiểm tra toàn instance rồi đưa candidate vào survival. Mọi full-score candidate vẫn cập nhật incumbent trước đó.",
        "Response lỗi thành no_op có reason. Lỗi cấu hình API ngừng review tiếp theo của instance; không âm thầm coi đó là quyết định hợp lệ.",
        "Local CP soft objective ưu tiên weighted time/room. Student/soft-distribution penalties được full scorer kiểm, chưa tối ưu trực tiếp đầy đủ trong CP.",
    ])
    pages.append(page)

    page=title("06  /  NGÂN SÁCH VÀ CẤU HÌNH","Cấu hình hiện chọn không phải CLI defaults")
    page+=table(["Setting","Main.py mặc định","Profile hiện chọn"],[
        ["Population / H","20 / 3","12 / 2 cho nbi, yach; 20 / 3 cho bốn instance khác"],
        ["Mutation rate / fraction","0,20 / 0,005","0,35 / 0,01"],
        ["Crossover","0,70","0,70"],
        ["Initial / offspring repair","12 steps, 3 pair trials / 8 steps, 2 pair trials","Giữ nguyên"],
        ["Inner directions","Tắt","Tắt; vẫn có tùy chọn CLI"],
        ["Seed mặc định","42","Wrapper dùng 17 nếu không truyền seed"],
    ],[.28,.26,.46])
    page+=[h("Caps theo instance trong profile")]
    rows=[]
    for name,conf in PROFILE["instances"].items():
        rows.append([name,str(conf["generations"]),f'{conf["evaluations"]:,}'.replace(",","."),
                     str(conf["time_limit_seconds"])])
    page+=table(["Instance","Gen cap","Eval cap","Wall cap (s)"],rows,[.43,.17,.18,.22])
    page+=bullets([
        "**Chỉ wrapper tự đọc profile:** dùng run_pilot.py. Gọi main.py trực tiếp dùng CLI defaults, trừ khi truyền đủ flags.",
        "Wrapper merge solver/repair overrides một lần và áp như nhau cho cả ba pipeline. So cùng instance phải dùng cùng profile và seed.",
        "Caps là hàng rào; run dừng khi một điều kiện chạm trước. Thêm repair evaluations khiến cùng evaluation cap không nhất thiết đạt cùng generation thực tế.",
        "Wall cap có hệ số an toàn, dự phòng generation dài và tối đa 360 giây API latency; không phải dự báo runtime baseline hay đo API thật.",
        "Preset chỉ được chọn trong miền đã thử. muni-fi/yach còn tiến triển ở evaluation cap; bet có endpoint bị time cap; recalibration_required vẫn true.",
    ])
    pages.append(page)

    page=title("07  /  CHẠY, BUDGET ACCOUNTING VÀ OUTPUT","Đọc đúng một lần chạy")
    page+=[h("Logical evaluations và thời gian")]
    page+=bullets([
        "ScoreEngine là LRU cache tối đa 16 chromosome/instance, giữ score và assignments. Cache hit vẫn tiêu logical evaluation; physical calls/cache hits ghi riêng.",
        "Search count cộng objective checks của Pymoo, routine/monitor repair, region scoring, operators và rechecks. Đối chiếu với score_engine.logical_requests.",
        "Local deltas/CP nodes chỉ tính runtime, không giả thành full evaluation. Hai final verification calls được liệt kê riêng ngoài search budget.",
        "Reserve allowance trước repair/controller; batch offspring cuối thu nhỏ để giữ evaluation cap. Deadline hợp tác: full score/initialization đã bắt đầu vẫn có thể hoàn tất sau wall cap.",
    ])
    page+=[h("Lệnh từ thư mục Experiment, trong venv đã cài requirements")]
    for cmd in ["python run_pilot.py --pipeline baseline --seed 17 --dry-run",
                "python run_pilot.py --pipeline baseline --seed 17",
                "python run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 42"]:
        page+=[p(cmd,"code")]
    page+=[p("Đổi pipeline thành single/multi và cấu hình model/API key để dùng LLM. Các lệnh agent có thể phát sinh chi phí API. Báo cáo này không thực thi các lệnh trên.","small")]
    page+=table(["Artifact","Dùng để đọc gì?"],[
        ["run_config.json / summary.json","Config, định danh run và tóm tắt các instance."],
        ["result.json","Score/components, evaluation breakdown, timing, sectioning và repair events."],
        ["choices.npz","Chromosome đã lưu để tái chấm hoặc probe riêng."],
        ["anytime_trace.jsonl","Tiến trình incumbent/population theo generation và budget."],
        ["controller_trace.jsonl","Trigger, quyết định, validation, outcome và chi phí agent."],
        ["solution.xml","Chỉ xuất khi nghiệm cuối hard 0 và sectioning qua kiểm tra nội bộ."],
    ],[.39,.61])
    page+=bullets([
        "Run ID gắn settings, XML/source hashes và dependency versions. Cùng định danh chỉ thay output sau khi run mới hoàn tất qua staging.",
        "Không có solution.xml không có nghĩa run thất bại kỹ thuật: xem hard, components, stop reason và trace trước.",
    ])
    pages.append(page)

    page=title("08  /  TRẠNG THÁI VÀ TRUY VẾT","Đã có gì; giới hạn nào vẫn còn?")
    page+=[h("Trạng thái đã ghi nhận, không phải cam kết cho mọi instance")]
    page+=bullets([
        "Đã có shared incumbent, group CP, forward-checking construction, adaptive/global CP, cache/guards, regional joint operator và capacity-plan sectioning fallback.",
        "Trong các cặp seed xác nhận, muni-fi, muni-fsps, mary và nbi đạt khả thi nội bộ. yach còn room clashes; bet còn lỗi phòng và required distributions.",
        "Missing student requests không gây hard còn lại ở các endpoint này. Room/group CP gates vẫn có thể bỏ qua lỗi hỗn hợp; ép CP riêng chưa giải được ca bet trong budget đã thử.",
        "Chưa chạy validator độc lập hoặc đánh giá LLM thật trong đợt calibration. Không coi UNKNOWN là vô nghiệm hay endpoint bị time cap là hội tụ.",
    ])
    page+=callout("Có thể bắt đầu test controller trước khi mọi instance đạt hard 0","Với instance chưa khả thi, theo dõi hard/components và tiến trình giảm hard. Khi đã khả thi, theo dõi thời điểm đạt hard 0 và soft quality. Giữ cùng solver/profile/seed; không trộn soft scores của nghiệm khả thi và chưa khả thi thành một kết luận.")
    page+=table(["Nhóm chức năng","Module chính trong NSGA-III/itc2019"],[
        ["Dữ liệu / construction","instance.py; construction.py"],
        ["Scoring / student assignment","scoring.py; sectioning.py"],
        ["NSGA-III / archive / routine repair","nsga3.py"],
        ["Monitor / CP / room search","controller.py; neighborhood.py; repair_policy.py; room_search.py"],
        ["StatePacket / actions / API","regions.py; operators.py; llm.py"],
        ["Run / kiểm tra tương thích","run_pilot.py; compare_pipelines.py (ở thư mục Experiment)"],
    ],[.38,.62])
    page+=[p("Giới hạn còn lại: search có sampled domains và bounded neighborhoods; capacity fallback có thể bỏ lỡ bundles; chưa có external Pareto archive; local CP chưa tối ưu trực tiếp mọi soft objective. Không khẳng định solver tối ưu toàn cục hoặc giải được toàn ITC 2019.","small")]
    pages.append(page)
    return pages


# Each record retains the original H identifier. Historical settings are not
# silently rewritten to the current profile; later replacement IDs are explicit.
HISTORY = [
("H01","Chuyển baseline sang ITC 2019",[
 "**Vấn đề:** solver cũ phụ thuộc dữ liệu và format riêng.",
 "**Thay đổi:** parser XML, option-based chromosome, scorer và solution writer theo cấu trúc ITC; thiết kế trên instance đầy đủ.",
 "**Quyết định:** subset là danh sách XML nguyên vẹn. Module/data format cũ không còn trong đường chạy.",
 "**Truy vết:** instance.py, scoring.py, main.py."]),
("H02","Output và định danh run theo cấu hình",[
 "**Vấn đề:** timestamp và chọn instance thủ công khiến khó tra kết quả.",
 "**Thay đổi:** --instance/--manifest/--all/--inspect; run ID gắn settings, XML, source và môi trường; lưu config, summary, choices và trace.",
 "**Quyết định:** cùng định danh chỉ ghi đè sau khi run mới hoàn tất qua staging; đổi seed/code/data/settings tạo run khác.",
 "**Truy vết:** main.py; hướng dẫn thao tác trong GUIDELINE.md."]),
("H03","Một solver chung cho ba pipeline",[
 "**Vấn đề:** bản sao baseline/agent có thể lệch thuật toán và repair.",
 "**Thay đổi:** hợp nhất parser, scorer, NSGA-III, repair và monitor trong itc2019; agent folders chỉ giữ entry point và vai trò riêng.",
 "**Quyết định:** bỏ solver copy riêng; mọi cải thiện core áp dụng cho cả ba pipeline.",
 "**Truy vết:** ba entry points; compare_pipelines.py kiểm solver/config/hash/dependencies."]),
("H04","Construction xét required distributions",[
 "**Vấn đề:** ưu tiên phòng/giờ có thể dựng lịch khó sửa vì bỏ qua required relationships.",
 "**Thay đổi:** index distributions theo class; khi nhóm đã được đặt đủ, thêm violations vào điểm chọn placement.",
 "**Giới hạn lúc đó:** chưa có forward checking trước khi đủ nhóm; vẫn là bounded heuristic, không phải exact CSP. H24 bổ sung phần này.",
 "**Truy vết:** _place trong nsga3.py; construction.py ở phiên bản mới."]),
("H05","Repair offspring tại chỗ",[
 "**Vấn đề:** dựng lại toàn timetable sau variation phá placement tốt và làm yếu crossover.",
 "**Thay đổi:** RoomRepair giữ chromosome offspring, dùng local hard repair quanh vi phạm.",
 "**Quyết định:** không biến repair thành construction mới. Lượt repair trùng sau initial sampling còn tồn tại lúc này, được bỏ ở H26.",
 "**Truy vết:** RoomRepair, PlacementSampling, _improve_hard trong nsga3.py."]),
("H06","Bounded two-class moves",[
 "**Vấn đề:** một class không có nước đi giảm hard nếu class liên quan chưa đổi cùng.",
 "**Thay đổi:** joint search lấy target từ violated distributions/room clashes; ưu tiên room domain hữu ích cho SameRoom.",
 "**Quyết định:** full scorer phải xác nhận giảm hard; hai-class search chưa giải quyết nhóm lớn. H23 mở rộng sang group CP.",
 "**Truy vết/kiểm tra:** _joint_improve_hard; regression improvement, accounting và deadline."]),
("H07","Acceptance trước và sau khả thi",[
 "**Vấn đề:** đòi mọi intervention đạt hard 0 loại bỏ bước cải thiện trên source chưa khả thi.",
 "**Thay đổi:** source hard > 0 chỉ cần giảm hard; source hard = 0 phải giữ khả thi và giảm soft total.",
 "**Quyết định:** áp trên bản sao, kiểm toàn instance; candidate vẫn qua survival. Quy tắc dùng chung cho hai controllers.",
 "**Truy vết/kiểm tra:** _admissible_candidate, apply_operator trong operators.py; acceptance regressions."]),
("H08","Tách khả thi khỏi điều kiện dừng",[
 "**Vấn đề:** tìm được lịch khả thi chưa có nghĩa hết cơ hội giảm preferences.",
 "**Thay đổi:** --stop-on-feasible là tùy chọn; mặc định tiếp tục tới cap.",
 "**Quyết định:** XML chỉ xuất cho nghiệm cuối khả thi/sectioning hợp lệ; không bắt solver dừng ngay khi có XML.",
 "**Truy vết/kiểm tra:** FeasibleSolutionTermination, run, main.py; test default tiếp tục sau khả thi."]),
("H09","Incumbent ban đầu và RNG selection",[
 "**Vấn đề:** nghiệm tốt có thể biến mất ở quần thể cuối; tie-breaking dùng sai RNG làm lệch seed.",
 "**Thay đổi:** monitor giữ best choices/quality, final đối chiếu population; comparator dùng random state của solver.",
 "**Giới hạn lúc đó:** archive theo (hard,total), survival vẫn đa mục tiêu; candidate từ operator chưa được ghi nhận thống nhất. H22 thay bằng shared incumbent.",
 "**Truy vết:** BaselineMonitor; _seeded_cv_tournament; final selection trong nsga3.py."]),
("H10","Room-chain qua trạng thái trung gian",[
 "**Vấn đề:** sửa clash có thể cần một chuỗi move mà bước đầu chưa giảm hard.",
 "**Thay đổi:** bounded beam/chain search, xét cả room unavailable.",
 "**Quyết định:** không đưa intermediate states vào population; chỉ trả candidate giảm hard và lỗi phòng. H15-H16 thu nhỏ và chuyển nhánh này thành fallback.",
 "**Truy vết:** _room_chain_repair trong nsga3.py."]),
("H11","Coordinated repair trong monitor",[
 "**Vấn đề:** distribution violations kéo dài qua nhiều vòng mutation/routine repair.",
 "**Thay đổi:** thử coordinated rounds định kỳ khi required violations không ít hơn room clashes.",
 "**Quyết định:** dùng chung cả ba pipeline, giữ deadline, scoring calls/events; ghi nhận cải thiện trước survival. H23 thêm group CP trước pair fallback.",
 "**Truy vết:** _maybe_coordinated_repair trong controller.py."]),
("H12","Room plateau search cho vùng hard cao",[
 "**Vấn đề:** incumbent room-heavy nhưng chưa gần khả thi nằm ngoài gate room-chain.",
 "**Thay đổi:** dùng local room/required deltas và neutral/worse intermediate moves trước full check; xét cả unavailability.",
 "**Quyết định:** bounded hard slack, số class đổi, deadline và full checks; chỉ nhận cải thiện cuối. Vai trò hiện tại khác room CP gần khả thi.",
 "**Truy vết:** room_search.py; _maybe_room_plateau_repair."]),
("H13","Scorer và chi phí đánh giá",[
 "**Vấn đề:** student sectioning lặp pair-conflict checks; thiếu stage timing; singleton MaxBlock cần sửa.",
 "**Thay đổi:** cache pair conflicts trong từng full score, thêm timing/candidate counters; bỏ penalty của block chỉ có một class.",
 "**Quyết định:** cache không biến greedy/beam thành exact sectioning; internal validation không phải độc lập. H26/H28 mở rộng cache và capacity fallback.",
 "**Truy vết:** scoring.py; PlacementProblem/sampling/repair trong nsga3.py."]),
("H14","Progress và diversity cho review",[
 "**Vấn đề:** chromosome khác nhau vẫn có thể gần giống về placement; soft dao động chưa nói lên hard progress.",
 "**Thay đổi:** placement diversity; progress theo hard trước khả thi, total sau khả thi; trigger một lần mỗi episode.",
 "**Quyết định:** uniqueness/reference-direction coverage vẫn log nhưng không làm low-diversity trigger hiện tại.",
 "**Truy vết:** placement_diversity/population_metrics trong regions.py; _progress_signal/_trigger trong controller.py."]),
("H15","Room CP mở sang blockers",[
 "**Vấn đề:** lớp không vi phạm đang chiếm option thay thế có thể chặn repair.",
 "**Thay đổi:** bounded neighborhood từ clashes/unavailability sang blockers; fixed outside; mã hóa room, pairwise required, MaxDays/MaxDayLoad.",
 "**Quyết định:** giữ incumbent option khi cắt domain, giới hạn model/memory/time. CP cục bộ không chứng minh vô nghiệm toàn instance; H25 thêm adaptive/global models.",
 "**Truy vết/kiểm tra:** neighborhood.py; _maybe_room_chain_repair; blocker/global-boundary regressions."]),
("H16","Chi phí repair và failure cooldown",[
 "**Vấn đề:** thử lặp repair đắt trên cấu trúc lỗi không đổi làm mất ngân sách.",
 "**Thay đổi:** preset 32 class/32 options, CP 8 giây; fallback 2 giây/128 checks; structural failure key, backoff 5 → 10 → 20 generations.",
 "**Quyết định:** retry theo lỗi/hard, không chỉ hash chromosome; cooldown không loại nghiệm. H25 cho phép tăng vùng trong deadline chung; H29 thêm override yach.",
 "**Truy vết:** RepairSettings/FailureBackoff trong repair_policy.py; controller.py."]),
("H17","Đếm đủ full evaluations và thêm cap",[
 "**Vấn đề:** chỉ đếm Pymoo bỏ sót routine/monitor repair và controller checks.",
 "**Thay đổi:** tổng search count gồm repair, regions, operators, rechecks; thêm --evaluations và remaining-budget guards; final checks tách riêng.",
 "**Quyết định:** CP nodes/local deltas chỉ tính runtime. Inner-loop guards và batch cap lúc này chưa đủ chặt; H26 bổ sung.",
 "**Truy vết:** evaluation_count, FullEvaluationTermination, regions.py, controller.py và main.py."]),
("H18","Setting làm việc ban đầu",[
 "**Vấn đề:** tăng population/H/repair không mặc nhiên tốt hơn dưới ngân sách hữu hạn.",
 "**Thay đổi:** chọn 20/H3/crossover 0,7 thay 40/H4/0,9; lúc này giữ mutation 0,2 / fraction 0,005, initial repair 12/3, offspring 8/2.",
 "**Quyết định:** chưa chọn H5, inner layer và các repair/mutation alternatives. Mở CLI để tuning; H29 chọn lại profile trên core mới, không đổi các CLI defaults này.",
 "**Truy vết:** Config/reference_directions, main.py, tune_solver.py."]),
("H19","Review gate và action limits",[
 "**Vấn đề:** checkpoint dày, events lặp và action nhỏ tốn review nhưng ít cơ hội sửa.",
 "**Thay đổi:** interval 5 → 10; min 2 → 3; max 10 → 20; recovery diversity 0,30 bên cạnh trigger 0,20; warm-up/window giữ 5.",
 "**Quyết định:** tối đa 6 regions, size 5, operator cap 24, reviews 12, timeout 15 giây. Chọn qua offline replay/action tests, chưa đánh giá LLM thật.",
 "**Truy vết:** ControllerSettings/_trigger/_packet; tune_review_gate.py; tune_controller.py."]),
("H20","Profile chạy local thống nhất",[
 "**Vấn đề:** wall cap chung không phù hợp mọi XML; lệnh thủ công dễ lệch role-call budgets.",
 "**Thay đổi:** pilot.json và run_pilot.py; caps theo instance; cùng solver/repair; single/multi tối đa một/hai calls mỗi review; đồng bộ OR-Tools requirements.",
 "**Quyết định:** dùng generation/evaluation/time caps. Budgets lúc đó còn provisional; calibration stopping rule không phải stop rule mặc định. H29-H30 đo lại.",
 "**Truy vết:** run_pilot.py, profiles/pilot.json, select_pilot_budget.py; ba requirements files."]),
("H21","Hợp nhất hai tài liệu chuẩn",[
 "**Vấn đề:** audit/calibration/comparison/progress documents chồng nhau và lệch implementation.",
 "**Thay đổi:** PIPELINES mô tả hiện trạng; IMPROVEMENT_HISTORY ghi quyết định; gỡ báo cáo số liệu cũ và sửa README links.",
 "**Quyết định:** không chép bảng benchmark hoặc thêm thời gian từng thay đổi; không xóa dataset/raw artifacts. R01-R08 khi ấy là việc mở, không phải thành tích đã làm.",
 "**Truy vết:** hai Markdown chuẩn; R01-R07 được triển khai tại H22-H28, R08 được đo lại tại H29-H30."]),
("H22","R01 - Shared incumbent trước survival",[
 "**Vấn đề:** monitor, controller/operator chưa cùng ghi nhận mọi candidate tốt.",
 "**Thay đổi:** PlacementProblem.score_state cập nhật Incumbent cho mọi đường full scoring; monitor/progress/feedback/output cùng dùng archive, tách population metrics.",
 "**Quyết định:** giữ một representative theo (hard,total), copy arrays để tránh mutation; không thay multi-objective survival, không gọi đây là Pareto archive. Thay giới hạn ở H09.",
 "**Truy vết/kiểm tra:** nsga3.py; _sync_incumbent trong controller.py; test candidate tốt bị survival loại nhưng archive còn đúng."]),
("H23","R02 - Required-distribution group repair",[
 "**Vấn đề:** single/two-class moves không vượt joint barrier; room-seeded CP không có target khi chỉ còn distribution lỗi.",
 "**Thay đổi:** distribution_neighborhood_repair lấy violated required group làm seed; joint finite-domain CP với fixed boundary, trước pair fallback.",
 "**Quyết định:** deadline chung 5 giây; chỉ yêu cầu giảm hard, không cần giảm room clash hoặc đạt hard 0. Mở rộng H06/H11, dùng chung cả ba pipeline.",
 "**Truy vết/kiểm tra:** neighborhood.py; _maybe_coordinated_repair; SameAttendees ba lớp không có room targets và saved-incumbent probe."]),
("H24","R03 - Forward-checking construction",[
 "**Vấn đề:** construction H04 chờ đủ nhóm mới xét distribution, có thể chặn lớp chưa đặt.",
 "**Thay đổi:** ConstructionIndex tái dùng required-pair cache; ưu tiên domain nhỏ/liên kết; partial global costs, tối đa 4 hàng xóm; giao SameRoom/SameStart domains.",
 "**Quyết định:** lấy mẫu 32 time/16 room, giữ 64 placements; seeded diversity. Hết deadline chỉ hoàn tất gene hợp domain, không gọi là đã repair; failure của sampled domains không chứng minh vô nghiệm.",
 "**Truy vết/kiểm tra:** construction.py; sampling trong nsga3.py; CommonStart regression và bounded probes trên sáu instance."]),
("H25","R04 - Adaptive neighborhood và global block CP",[
 "**Vấn đề:** trần vùng cố định bỏ sót blockers; MaxBreaks/MaxBlock chỉ lộ qua full check.",
 "**Thay đổi:** 16/16 → preset 32/32 → tối đa 64/64; thêm fixed-boundary MaxBreaks/MaxBlock, gom day/week patterns, đếm đúng duplicate intervals và singleton.",
 "**Quyết định:** một deadline/check budget cho các tầng, không nhân timeout; model lớn/hết budget không suy ra instance vô nghiệm. Thay giới hạn H15-H16; giữ fallback/backoff.",
 "**Truy vết/kiểm tra:** neighborhood.py; exhaustive fixture CP-vs-scorer cho bốn global constraints, fixed boundary, nhiều ngày/tuần và duplicate intervals."]),
("H26","R05 - Scoring cache và budget guards",[
 "**Vấn đề:** rechecks lặp sectioning; initial repair bị gọi hai lần; cap/deadline thiếu guard bên trong.",
 "**Thay đổi:** ScoreEngine LRU 16 entries giữ score/assignments; hit vẫn tính logical evaluation; physical/cache counters; inner guards, product-domain threshold 96; bỏ repair trùng; thu nhỏ offspring batch cuối.",
 "**Quyết định:** reserve objective/recheck allowance; local delta không giả full score; final checks riêng, time cap vẫn hợp tác. Không tuyên bố incremental student scoring giữa hai chromosomes. Mở rộng H05/H13/H17.",
 "**Truy vết/kiểm tra:** scoring.py, nsga3.py, main.py và helpers; cache/deadline/non-divisible-cap tests; đối chiếu evaluation count cho baseline và fake controllers."]),
("H27","R06 - Local diagnosis và joint regional operator",[
 "**Vấn đề:** representatives theo total ít trade-offs; attribution thô; destroy/repair ba lớp tuần tự mắc joint barrier.",
 "**Thay đổi:** representative khác ưu tiên objective/placement distance; connected regions, đúng violated endpoints, student edges từ assignment; boundary giữ relationships/blockers. CP sửa đồng thời cả vùng rồi fallback nếu còn budget.",
 "**Quyết định:** không mở target ra ngoài vùng agent đã chọn; CP tối đa 3 giây, chung allowance. Public boundary cắt 32 IDs có total/truncated; full boundary giữ nội bộ. Global attribution vẫn heuristic.",
 "**Truy vết/kiểm tra:** regions.py, operators.py; fixture bốn lớp phải đổi cùng, fixed outside, archive qua survival reject và endpoint attribution."]),
("H28","R07 - Capacity-plan sectioning fallback",[
 "**Vấn đề:** greedy chiếm capacity cần cho người sau; beam 12 có thể bỏ parent choice hợp lệ duy nhất.",
 "**Thay đổi:** khi còn unassigned, sinh parent-topological bundles; CP số lượng người mỗi bundle; gán lại toàn bộ course lỗi, ghép theo student histories và tính lại conflicts/missing.",
 "**Quyết định:** cache structural plan, giới hạn 4.096 bundles/200.000 nodes; CP một worker, seed 0, det limit 0,5. Giữ greedy khi gán đủ; không bảo đảm tối ưu hay đầy đủ bundles.",
 "**Truy vết/kiểm tra:** sectioning.py, scoring.py, instance cache; capacity reassignment, parent ngoài beam, deterministic repeats và true-capacity-shortage tests. Đồng bộ source fingerprints; caps cũ đánh dấu cần đo lại."]),
("H29","R08 - Chọn lại cấu hình trên core đã tăng cường",[
 "**Vấn đề:** cấu hình H18-H20 được chọn trước thay đổi construction, CP, cache và sectioning.",
 "**Cách đo:** giữ nguyên core; sàng lọc sáu cấu hình, promotion trên ca khó/seed 73, thử mutation-population combinations, CP budgets và xác nhận sáu instance ở seed 17/42. Ghi config, hashes, môi trường, trajectory và repair events.",
 "**Chọn:** profile mutation 0,35 / fraction 0,01; nbi/yach population 12/H2, bốn file khác 20/H3; giữ crossover 0,7, repair 12/3 và 8/2, không inner layer. Room CP mặc định 8 giây/det 0,5; yach 15 giây/det 2 sau xác nhận hai seed.",
 "**Không chọn:** hai lớp directions, repair mạnh hơn, vùng rộng hơn hoặc room gate hard 40 chưa đủ ổn định. Gate 40 còn thay nhánh được chạy, không chỉ cấp thêm compute.",
 "**Thực thi/kiểm tra:** profile thay đổi, CLI defaults giữ nguyên; wrapper merge overrides một lần cho cả ba pipeline. Regression cho overrides/seed consistency/censoring; warm CP cho unit deadlines, calibration vẫn tính cold start. Không gọi API hay sửa dataset/core.",
 "**Truy vết:** recalibrate_solver.py; analyze_recalibration.py; run_pilot.py; accepted-record manifest và replay manifests có source/environment guards."]),
("H30","Working caps và phân loại nút thắt còn lại",[
 "**Vấn đề:** smoke run/một seed không tách được thiếu budget và repair bottleneck; time cap không phải hội tụ.",
 "**Cách chọn eval cap:** mốc 90% observed feasible soft gain cộng đệm và ít nhất 400 evaluations sau khả thi; nếu chưa khả thi, lần giảm hard cuối cộng đệm 400. Làm tròn 100, không vượt horizon đo chung hai seed.",
 "**Cách chọn wall cap:** thời gian tới cap của seed chậm hơn × 1,35, cộng hai generation dài và tối đa 360 giây API reserve. Đây là safety ceiling từ job kỹ thuật đồng thời, không phải phép so tốc độ hay đo API.",
 "**Trạng thái:** muni-fi/yach còn progressing_at_evaluation_cap; bet time_censored; mary/muni-fsps/nbi có measured_local_knee trong phạm vi đo. Giữ recalibration_required = true.",
 "**Chẩn đoán:** bốn instance khả thi nội bộ; yach còn room clashes, bet có lỗi phòng và required distributions; missing requests không gây hard còn lại. Mixed conflicts có thể không mở cả hai CP gates.",
 "**Probe/giới hạn:** probe chạy riêng trên saved choices, không cộng điểm probe vào GA. yach sửa được một bước; bet room/group CP vẫn chưa tìm được cải thiện trong budget. UNKNOWN/local INFEASIBLE không chứng minh XML vô nghiệm. Validator/LLM thật chưa thực hiện.",
 "**Truy vết:** budget suggestion, accepted-record manifest, profile backup và raw JSON/NPZ; probe_neighborhood.py, check_strengthening.py. Giữ hai tài liệu chuẩn, không phục hồi các báo cáo số liệu cũ."]),
]


def entry(item):
    code,name,body=item
    return [h(code+". "+name),*bullets(body,"history"),Spacer(1,3)]


def history_pages():
    pages=[]
    page=title("BÁO CÁO 02  /  LỊCH SỬ VÀ QUYẾT ĐỊNH","Lịch sử cải thiện solver","30 mốc thay đổi, quyết định và kiểm chứng kỹ thuật; nhóm theo giai đoạn phát triển.",True)
    page+=fig("history_roadmap",305,"Hình 1. Sáu giai đoạn phát triển theo thứ tự H. Đây là sơ đồ tổ chức lịch sử, không phải lịch thời gian và không thể hiện mức tăng hiệu năng.")
    page+=bullets([
        "**Mã H là mục tài liệu**, không phải Git commit. Mỗi mục giữ vấn đề, thay đổi, quyết định/giới hạn và điểm truy vết.",
        "**Đọc theo phiên bản:** setting hay giới hạn ở một mốc cũ mô tả đúng trạng thái lúc đó. Nếu có thay thế, ưu tiên mốc mới hơn và bản mô tả solver hiện tại.",
        "**R01-R07:** các phần mở ở H21 đã triển khai tại H22-H28. **R08:** đo lại ở H29-H30 nhưng một số horizon/budget vẫn còn mở.",
        "**Không thêm kết quả mới:** tài liệu này không chạy benchmark, không bổ sung API calls, không dựng lại các báo cáo số liệu đã gỡ.",
    ])
    pages.append(page)

    groups=[
        ("01  /  NỀN TẢNG","Từ format riêng đến shared solver",1,4),
        ("02  /  CƠ CHẾ SEARCH","Giữ cấu trúc tốt và ghi nhận tiến bộ",5,9),
        ("03  /  VƯỢT PLATEAU","Room search, coordinated repair và scorer",10,13),
        ("04  /  ĐIỀU KHIỂN CHI PHÍ","Review signals, CP và evaluation cap",14,17),
        ("05  /  CHUẨN HÓA","Setting, local profile và tài liệu",18,21),
        ("06  /  TĂNG CƯỜNG LÕI - A","Archive, group repair và construction",22,24),
        ("07  /  TĂNG CƯỜNG LÕI - B","Global CP, cache và sectioning",25,28),
    ]
    for label,heading,first,last in groups:
        page=title(label,heading)
        for item in HISTORY[first-1:last]:
            page+=entry(item)
        pages.append(page)

    page=title("08  /  HIỆU CHỈNH LẠI","Hiệu chỉnh cấu hình trên solver mới")
    page+=entry(HISTORY[28])
    page+=entry(HISTORY[29])
    pages.append(page)

    page=title("09  /  ĐỌC NHANH QUYẾT ĐỊNH","Những mốc nào thay đổi quyết định trước?")
    page+=fig("history_dependencies",290,"Hình 2. Các quan hệ bổ sung/thay thế chính. Không xóa nội dung mốc cũ hoặc viết lại như thể implementation mới đã luôn tồn tại.")
    page+=table(["Chủ đề","Mốc gốc → bổ sung / thay thế"],[
        ["Acceptance / stop","H07-H08 vẫn giữ: giảm hard trước khả thi; hard 0 không phải stop bắt buộc."],
        ["Construction / incumbent","H04 → H24; H09 → H22."],
        ["Joint / room repair","H06/H11 → H23; H10/H15/H16 → H25 và overrides H29."],
        ["Scoring / sectioning / budget","H13/H17 → H26/H28; H20 → H29-H30."],
        ["Setting / tài liệu","H18 được chọn lại trong profile H29; CLI defaults vẫn khác profile. H21 giữ hai Markdown chuẩn."],
    ],[.29,.71])
    page+=callout("Trạng thái hiện tại cần giữ nguyên khi đọc lịch sử","Có implementation và regression cho R01-R07; preset đã được đo lại nhưng chưa giải hết ca khó. Test controller không bắt buộc mọi instance phải hard 0; validator độc lập và LLM thật vẫn là phần chưa thực hiện.")
    page+=[p("Nguồn biên tập: toàn bộ IMPROVEMENT_HISTORY.md (H01-H30), đối chiếu PIPELINES.md và profiles/pilot.json. Nội dung đã được rút gọn để đọc theo quyết định; Markdown gốc vẫn giữ nguyên cho truy vết chi tiết.","small")]
    pages.append(page)
    return pages


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    QA.mkdir(parents=True,exist_ok=True)
    source_paths=[ROOT/"PIPELINES.md",ROOT/"IMPROVEMENT_HISTORY.md",ROOT/"profiles/pilot.json"]
    manifest={"sources":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in source_paths},
              "history_ids":[item[0] for item in HISTORY],
              "figures":"Implementation schematics; region graph illustrative, not experimental data.",
              "core_solver_changed":False,"new_experiments":False,
              "expected_pages":{"SOLVER_CURRENT_REPORT.pdf":9,"SOLVER_IMPROVEMENT_HISTORY.pdf":10}}
    assert manifest["history_ids"]==[f"H{i:02d}" for i in range(1,31)]
    for file in [build("SOLVER_CURRENT_REPORT.pdf",pipeline_pages(),"Solver hiện tại"),
                 build("SOLVER_IMPROVEMENT_HISTORY.pdf",history_pages(),"Lịch sử cải thiện")]:
        print(file)
    (QA/"report_build_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
