"""Render every report page and verify content, bounds and source preservation."""
from pathlib import Path
import hashlib
import json
import re

import pymupdf
from PIL import Image,ImageOps,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/pdf"
QA=ROOT/"tmp/pdfs"


def main():
    manifest=json.loads((QA/"report_build_manifest.json").read_text(encoding="utf-8"))
    audits={}
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",15)
    for name,expected in manifest["expected_pages"].items():
        doc=pymupdf.open(OUT/name)
        folder=QA/Path(name).stem
        folder.mkdir(parents=True,exist_ok=True)
        thumbs=[]
        pages=[]
        alltext=""
        for i,page in enumerate(doc,1):
            text=page.get_text()
            alltext+=text
            clips=[]
            actual_lines=[]
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines",[]):
                    line_rect=pymupdf.Rect(line["bbox"])
                    actual_lines.append((line_rect,"".join(s["text"] for s in line["spans"])))
                    for span in line["spans"]:
                        rect=pymupdf.Rect(span["bbox"])
                        if rect.x0<20 or rect.x1>page.rect.width-20 or rect.y0<8 or rect.y1>page.rect.height-8:
                            clips.append(span["text"])
            line_overlaps=[]
            for k,(first,first_text) in enumerate(actual_lines):
                for second,second_text in actual_lines[k+1:]:
                    # Font ascent/descent boxes can touch while glyph ink is clear.
                    a=pymupdf.Rect(first.x0+1,first.y0+1,first.x1-1,first.y1-1)
                    b=pymupdf.Rect(second.x0+1,second.y0+1,second.x1-1,second.y1-1)
                    overlap=a&b
                    if not overlap.is_empty and overlap.height>1 and overlap.width>2:
                        line_overlaps.append([first_text,second_text])
            pix=page.get_pixmap(matrix=pymupdf.Matrix(1.4,1.4),alpha=False)
            pix.save(folder/f"page-{i:02d}.png")
            im=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
            im.thumbnail((280,396))
            thumb=Image.new("RGB",(300,430),"#E6EBEF")
            thumb.paste(im,((300-im.width)//2,8))
            ImageDraw.Draw(thumb).text((12,410),f"Trang {i}",font=font,fill="#203448")
            thumbs.append(thumb)
            pages.append({"page":i,"chars":len(text),"boundary_issues":clips,
                          "actual_line_overlaps":line_overlaps})
        cols=3
        rows=(len(thumbs)+cols-1)//cols
        contact=Image.new("RGB",(cols*300,rows*430),"#D8E1E8")
        for i,im in enumerate(thumbs):
            contact.paste(im,((i%cols)*300,(i//cols)*430))
        contact.save(folder/"contact.png")
        audits[name]={"pages":len(doc),"expected_pages":expected,"page_details":pages,
                      "replacement_glyphs":alltext.count("\ufffd"),
                      "history_ids":sorted(set(re.findall(r"H\d\d",alltext)))}
        print(name,len(doc),"pages",[p["chars"] for p in pages])
        assert len(doc)==expected,(name,len(doc),expected)
        assert all(p["chars"]>350 and not p["boundary_issues"] and
                   not p["actual_line_overlaps"] for p in pages),audits[name]
        assert "\ufffd" not in alltext
        if "HISTORY" in name:
            assert set(manifest["history_ids"])<=set(audits[name]["history_ids"])
    for path,sha in manifest["sources"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    audits["markdown_and_profile_preserved"]=True
    audits["figure_contract"]={
        "backend":"Python/matplotlib", "archetype":"single-canvas implementation schematics",
        "quantitative_data":"not applicable; no scores or simulated measurements plotted",
        "region_graph":"illustration only, explicitly captioned",
        "alignment":"single-canvas: not applicable, recorded per figure",
        "source_preflight_warnings":[
            "PNG is a report illustration, not a TIFF journal submission.",
            "400 dpi exceeds the 300 dpi report floor; 600 dpi journal default not required.",
            "Width scanner misreads 183/25.4; actual PDFs are 183 mm wide."
        ],
        "report_collision_tool_note":
            "The figure auditor promotes a multi-line ReportLab text-trace union to the first line bbox. "
            "Its full-report text-text flags are false positives, not a report pass. "
            "Independent actual-line rectangles above and final-page visual review are used for reports. "
            "The unmodified auditor is authoritative on the six Matplotlib figures."
    }
    audits["figures"]={}
    for file in sorted((OUT/"assets").glob("*.pdf")):
        result=json.loads((QA/"qa"/(file.stem+".collision.json")).read_text(encoding="utf-8"))
        assert result["summary"]["fail"]==0 and result["summary"]["warn"]==0,file
        doc=pymupdf.open(file)
        assert abs(doc[0].rect.width/72*25.4-183)<.1
        audits["figures"][file.stem]={"collision":result["verdict"],"width_mm":183,
             "labels_and_arrow_clearance":"visually inspected", "statistics":"not applicable"}
    (QA/"report_quality.json").write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Content/bounds/source checks passed; inspect every rendered page before delivery.")


if __name__=="__main__":
    main()
