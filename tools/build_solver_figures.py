"""Technical diagrams for the two solver reports; no simulated observations.

Figure contract: explain execution, bounded repair, and revision dependencies.
Archetype: single-canvas schematics, not quantitative or comparable-panel plots.
Backend: Python/matplotlib only. Export: 183 mm, editable PDF/SVG, 400 dpi PNG.
All diagrams describe the implementation; the region graph is illustrative.
"""
from pathlib import Path
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

SKILL = Path("C:/Users/black/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/assets"
QA = ROOT / "tmp/pdfs/qa"
BLUE, TEAL, INK, MUTED = "#254F73", "#137F83", "#203448", "#586B7A"
LIGHT, PALE, AMBER = "#F0F5F9", "#EAF6F4", "#A26728"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})


def canvas(height):
    fig, ax = plt.subplots(figsize=(183/25.4, height/25.4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, s, size=9, weight="normal", color=INK, ha="center"):
    return ax.text(x, y, s, ha=ha, va="center", fontsize=size,
                   weight=weight, color=color, linespacing=1.35)


def box(ax, x, y, w, h, title, detail="", accent=False, size=9):
    color = TEAL if accent else BLUE
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=1.5", linewidth=0.9,
        edgecolor=color, facecolor=PALE if accent else LIGHT))
    if detail:
        text(ax, x+w/2, y+h*.65, title, size, "bold", color)
        text(ax, x+w/2, y+h*.28, detail, max(7.8, size-.6))
    else:
        text(ax, x+w/2, y+h/2, title, size, "bold", color)


def arrow(ax, points, color=BLUE, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.0,
                linestyle="--" if dashed else "-", solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
        mutation_scale=11, linewidth=1.1, color=color,
        linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0,
        joinstyle="miter"))


def save(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,
        json_out=str(QA / (name+".alignment.json")),
        overlay_svg=str(QA / (name+".alignment.svg")), strict=True)
    fig.savefig(OUT / (name+".pdf"))
    fig.savefig(OUT / (name+".svg"))
    fig.savefig(OUT / (name+".png"), dpi=400)
    plt.close(fig)


def solver():
    fig, ax = canvas(108)
    box(ax, 4, 79, 26, 17, "ITC 2019 XML", "Parser + option domains")
    box(ax, 37, 79, 26, 17, "Construction", "Forward checking + repair")
    box(ax, 70, 79, 26, 17, "Population đầu", "Full scoring + sectioning")
    arrow(ax, [(30, 87.5), (37, 87.5)])
    arrow(ax, [(63, 87.5), (70, 87.5)])
    arrow(ax, [(83, 79), (83, 70)])
    box(ax, 70, 52, 26, 18, "Variation + repair", "Crossover / mutation")
    box(ax, 37, 52, 26, 18, "Score + survival", "NSGA-III, 4 objectives")
    box(ax, 4, 52, 26, 18, "Repair dùng chung", "Distribution / room / CP")
    arrow(ax, [(70, 61), (63, 61)])
    arrow(ax, [(37, 61), (30, 61)])
    box(ax, 4, 25, 26, 17, "Controller tùy chọn", "Review → operator hợp lệ", True)
    arrow(ax, [(17, 52), (17, 42)])
    arrow(ax, [(30, 33.5), (83, 33.5), (83, 52)])
    text(ax, 57, 38.5, "Còn ngân sách: thế hệ tiếp theo", 8.4)
    box(ax, 38, 4, 58, 17, "Incumbent + kiểm tra cuối + output", "Mọi full-score candidate đều được ghi nhận")
    arrow(ax, [(17, 25), (17, 12.5), (38, 12.5)])
    text(ax, 27, 18.5, "Chạm cap", 8)
    save(fig, "solver_overview")


def region():
    fig, ax = canvas(72)
    ax.add_patch(FancyBboxPatch((5, 18), 46, 67, boxstyle="round,pad=0,rounding_size=2",
                             facecolor=PALE, edgecolor=TEAL, lw=1.2))
    text(ax, 28, 76, "VÙNG ĐƯỢC CHỌN", 9, "bold", TEAL)
    nodes = {"C1":(17,49), "C2":(39,49), "C3":(28,30), "C4":(72,64), "C5":(72,32)}
    for a,b in [("C1","C2"),("C2","C3"),("C2","C4"),("C3","C5")]:
        p,q=nodes[a],nodes[b]
        dx,dy=q[0]-p[0],q[1]-p[1]
        length=math.hypot(dx,dy)
        offset=5.8
        ax.plot([p[0]+offset*dx/length,q[0]-offset*dx/length],
                [p[1]+offset*dy/length,q[1]-offset*dy/length],
                color=TEAL if b in ("C2","C3") else MUTED,
                lw=1.3, linestyle="-" if b in ("C2","C3") else "--")
    for label,(x,y) in nodes.items():
        ax.add_patch(Circle((x,y),5.3,facecolor="white",edgecolor=TEAL if x<50 else BLUE,lw=1.2))
        text(ax,x,y,label,9,"bold")
    text(ax,72,84,"BOUNDARY",9,"bold",BLUE)
    text(ax,85,63,"Quan hệ\nràng buộc",8)
    text(ax,85,31,"Lớp chặn\nphương án khác",8)
    text(ax,28,8,"Được phép đổi placement",8.6,color=TEAL)
    text(ax,75,8,"Giữ cố định; vẫn kiểm tra",8.6)
    save(fig,"region_boundary")


def repairs():
    fig,ax=canvas(97)
    rows=[
        (76,"Required distributions > 0\nvà ≥ room clashes","Group CP + pair fallback","Deadline chung 5 giây"),
        (51,"Hard > 12; lỗi phòng ≥ 4","Room plateau search","Neutral / worse intermediate moves"),
        (26,"Hard ≤ 12; còn lỗi phòng\ninterval / backoff cho phép","Adaptive room CP → beam","16/16 → 32/32 → tối đa 64/64"),
    ]
    text(ax,25,97,"ĐIỀU KIỆN CHÍNH",9,"bold",BLUE)
    text(ax,76,97,"NHÁNH REPAIR",9,"bold",BLUE)
    for y,gate,title,detail in rows:
        box(ax,4,y,42,17,gate,size=8.7)
        box(ax,56,y,40,17,title,detail,True,size=8.7)
        arrow(ax,[(46,y+8.5),(56,y+8.5)])
    text(ax,50,15,"Các guard được xét theo thứ tự; không phải ba nhánh loại trừ hoàn toàn.",8.1)
    text(ax,50,6,"Routine repair luôn thuộc construction / variation; full scorer xác nhận kết quả.",8.1)
    save(fig,"repair_routes")


def controllers():
    fig,ax=canvas(84)
    text(ax,3,94,"SINGLE-AGENT",9,"bold",BLUE,ha="left")
    box(ax,3,66,24,19,"StatePacket","Progress + regions")
    box(ax,37,66,26,19,"Một agent","WHERE + HOW + WHEN",True)
    box(ax,73,66,24,19,"Validation","Operator + feedback")
    arrow(ax,[(27,75.5),(37,75.5)])
    arrow(ax,[(63,75.5),(73,75.5)])
    text(ax,3,51,"INSPECTOR-PLANNER",9,"bold",BLUE,ha="left")
    box(ax,3,20,19,20,"StatePacket","Cùng đầu vào",size=8.8)
    box(ax,29,20,19,20,"Inspector","WHERE",True,size=8.8)
    box(ax,57,20,19,20,"Planner","HOW + WHEN",True,size=8.8)
    box(ax,83,20,14,20,"Execute","Kiểm tra",size=8.5)
    arrow(ax,[(22,30),(29,30)])
    arrow(ax,[(48,30),(57,30)])
    text(ax,52.5,44,"Check target",7.9)
    arrow(ax,[(76,30),(83,30)])
    text(ax,50,7,"Cùng solver, operators, acceptance và ngân sách; chỉ khác cách phân vai.",8.4)
    save(fig,"controller_roles")


def evolution():
    fig,ax=canvas(110)
    stages=[
        (4,73,"01  Nền tảng chung","H01-H03","ITC XML / run identity / shared solver"),
        (55,73,"02  Search + acceptance","H04-H09","Construction / joint moves / incumbent"),
        (55,41,"03  Repair + đo chi phí","H10-H17","Room search / CP / full evaluations"),
        (4,41,"04  Cấu hình + tài liệu","H18-H21","Parameters / triggers / shared profile"),
        (4,9,"05  Tăng cường lõi","H22-H28","Archive / group CP / cache / sectioning"),
        (55,9,"06  Hiệu chỉnh lại","H29-H30","Config / working caps / unresolved cases"),
    ]
    for x,y,title,ids,detail in stages:
        box(ax,x,y,41,23,title,ids,accent=y==9,size=9.1)
        text(ax,x+20.5,y-5,detail,7.8)
    arrow(ax,[(45,84.5),(55,84.5)])
    arrow(ax,[(96,84.5),(99,84.5),(99,52.5),(96,52.5)])
    arrow(ax,[(55,52.5),(45,52.5)])
    arrow(ax,[(4,52.5),(1,52.5),(1,20.5),(4,20.5)])
    arrow(ax,[(45,20.5),(55,20.5)])
    save(fig,"history_roadmap")


def replacements():
    fig,ax=canvas(105)
    rows=[(79,"H04  Construction ban đầu","H24  Forward checking"),
          (61,"H09  Archive tại monitor","H22  Incumbent dùng chung"),
          (43,"H15-H16  Vùng CP cố định","H25  Vùng tăng dần + global models"),
          (25,"H13 / H17  Scoring + cap","H26 / H28  Cache, guards, capacity"),
          (7,"H18 / H20  Setting cũ","H29-H30  Preset đo trên lõi mới")]
    text(ax,25,98,"QUYẾT ĐỊNH TRƯỚC",9,"bold",BLUE)
    text(ax,76,98,"MỐC BỔ SUNG / THAY THẾ",9,"bold",TEAL)
    for y,old,new in rows:
        box(ax,3,y,43,12,old,size=8.5)
        box(ax,56,y,41,12,new,accent=True,size=8.4)
        arrow(ax,[(46,y+6),(56,y+6)])
    save(fig,"history_dependencies")


if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    QA.mkdir(parents=True,exist_ok=True)
    for build in (solver,region,repairs,controllers,evolution,replacements):
        build()
    print("Created six implementation diagrams (PDF/SVG/PNG).")
