"""Official-response bookkeeping tests, not an independent ITC scorer."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.validate_itc2019 import compare_response,submit,expected_for

EXPECTED={"instance":"fixture","classes":2,"score":{"hard":0,"total":120}}
OK={"obj":{"result":"OK","instance":"fixture","assignedVariables":{"value":2,"total":2},
           "totalCost":{"value":120},"timePenalty":{"value":1},"log":[]}}


def test_compares_official_total_not_just_ok_label():
    assert compare_response(OK,EXPECTED)["official_status"]=="passed"
    changed=deepcopy(OK)
    changed["obj"]["totalCost"]["value"]=121
    assert compare_response(changed,EXPECTED)["official_status"]=="penalty_mismatch"


@pytest.mark.parametrize("raw,status",[
    ({"obj":{"result":"NOT-VALID"}},"not_valid"),
    ({"obj":{"result":"NOT-SOLUTION"}},"custom_problem_not_competition_validation"),
    ({"obj":{"result":"OK"}},"instance_mismatch"),
    ({"obj":{"result":"OK","instance":"fixture"}},"unknown_response"),
    ({},"unknown_response"),
    ([],"unknown_response"),
])
def test_unknown_or_custom_results_are_not_official_pass(raw,status):
    assert compare_response(raw,EXPECTED)["official_status"]==status


def test_incomplete_assignment_not_passed():
    raw=deepcopy(OK)
    raw["obj"]["assignedVariables"]["value"]=1
    assert compare_response(raw,EXPECTED)["official_status"]=="incomplete_assignment"


def test_nonfinite_penalty_is_not_passed():
    raw=deepcopy(OK)
    raw["obj"]["totalCost"]["value"]="NaN"
    assert compare_response(raw,EXPECTED)["official_status"]=="unknown_response"


def test_missing_login_makes_no_network_request(monkeypatch,tmp_path):
    monkeypatch.delenv("ITC2019_TOKEN",raising=False)
    result=submit(tmp_path/"not-even-read.xml")
    assert result["official_status"]=="auth_required"
    assert result["network_requests"]==0 and result["published"] is False


def test_hash_binding_rejects_different_xml(tmp_path):
    solution=tmp_path/"solution.xml"
    solution.write_text('<solution name="fixture"/>')
    expected=tmp_path/"expected.json"
    expected.write_text(json.dumps({**EXPECTED,"solution_sha256":"old"}))
    with pytest.raises(ValueError,match="hash mismatch"):
        expected_for(solution,expected)
