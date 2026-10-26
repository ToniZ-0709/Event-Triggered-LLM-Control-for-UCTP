"""Portable geometry audit for the single-row grids used by search figures."""
import json
from pathlib import Path


def require_matplotlib_panel_alignment(fig, *, json_out, tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5,
                                       require_panel_labels=False, strict=True):
    fig.canvas.draw()
    width, height = fig.get_size_inches()*72
    rectangles = []
    for ax in fig.axes:
        x,y,w,h = ax.get_position().bounds
        rectangles.append([x*width,y*height,w*width,h*height])
    errors = []
    if len(rectangles) > 1:
        for column,name in ((1,"bottom"),(2,"width"),(3,"height")):
            values = [r[column] for r in rectangles]
            if max(values)-min(values) > tolerance_pt:
                errors.append("Unequal panel " + name)
        ordered = sorted(rectangles)
        gutters = [b[0]-a[0]-a[2] for a,b in zip(ordered,ordered[1:])]
        if any(g < 0 for g in gutters):
            errors.append("Overlapping plot areas")
        if gutters and max(gutters)-min(gutters) > gutter_tolerance_pt:
            errors.append("Unequal gutters")
    report = {"scope":"final rendered single-row plot areas", "rectangles_pt":rectangles,
              "tolerance_pt":tolerance_pt,"errors":errors,"passed":not errors}
    Path(json_out).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if errors and strict:
        raise ValueError("; ".join(errors))
    return report
