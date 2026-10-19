"""CLI/staging integration on a tiny synthetic fixture; no external requests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from itc2019.controller import ControllerCallback
from itc2019 import llm


@pytest.mark.parametrize("api_failure",[False,True])
def test_cli_publishes_bound_xml_or_incomplete_api_run(tmp_path,monkeypatch,api_failure):
    dataset=tmp_path/"data"
    dataset.mkdir()
    option='<room id="1" penalty="0"/><room id="2" penalty="0"/>'
    times='<time days="1" weeks="1" start="0" length="1" penalty="0"/>' \
          '<time days="1" weeks="1" start="1" length="1" penalty="0"/>'
    (dataset/"tiny.xml").write_text(
        '<problem name="tiny" nrDays="1" nrWeeks="1" slotsPerDay="4">'
        '<optimization time="1" room="1" distribution="1" student="1"/>'
        '<rooms><room id="1"/><room id="2"/></rooms><courses>'+
        ''.join(f'<course id="{i}"><config id="{i}"><subpart id="{i}">'
                f'<class id="{i}" limit="10">{option}{times}</class>'
                '</subpart></config></course>' for i in (1,2))+
        '</courses><distributions/><students/></problem>')
    spec=importlib.util.spec_from_file_location("tested_baseline_cli",BASE/"main.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output=tmp_path/"output"
    args=["main.py","--dataset",str(dataset),"--instance","tiny","--output",str(output),
          "--population","4","--partitions","1","--generations","8",
          "--initial-repair-steps","0","--offspring-repair-steps","0"]
    kwargs={}
    if api_failure:
        args += ["--warmup-generations","2","--review-interval","2","--min-review-interval","1"]
        class QuotaController(ControllerCallback):
            def decide(self,packet):
                return None,[{"api_requests":1,"status":"api_quota_exhausted"}],"api_quota_exhausted"
        monkeypatch.setattr(llm,"GeminiDecisionClient",lambda *a,**kw:
                            type("OfflineClient",(),{"metadata":lambda self:{"provider":"gemini"},
                                                      "close":lambda self:None})())
        kwargs={"controller_factory":QuotaController,"pipeline_name":"test-agent"}
    monkeypatch.setattr(sys,"argv",args)
    if api_failure:
        with pytest.raises(SystemExit) as exc:
            module.main(**kwargs)
        assert exc.value.code==3
    else:
        module.main(**kwargs)
    runs=list(output.glob("run_*"))
    assert len(runs)==1 and not list(output.glob(".*.tmp-*"))
    manifest=json.loads((runs[0]/"run_config.json").read_text())
    summary=json.loads((runs[0]/"summary.json").read_text())[0]
    assert manifest["completion_status"]==("incomplete_api_failure" if api_failure else "completed")
    if api_failure:
        assert summary["controller"]["comparison_eligible"] is False
        assert summary["controller"]["api_calls"]==1
    solution=runs[0]/"tiny/solution.xml"
    assert solution.exists()
    expected=json.loads((solution.parent/"expected.json").read_text())
    assert expected["solution_sha256"]==hashlib.sha256(solution.read_bytes()).hexdigest()
    assert expected["official_status"]=="not_run" and expected["score"]["hard"]==0
