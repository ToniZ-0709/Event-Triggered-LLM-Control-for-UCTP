"""Prepare existing feasible choices and compare official validation responses.

prepare: local re-score/export only, no GA optimization and no network.
submit: POST only to the site's validation endpoint, never its publication API.
import: compare a response saved manually through the official website.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"NSGA-III"))
from itc2019.instance import load_instance
from itc2019.scoring import evaluate,validate_sectioning

ENDPOINT="https://www.itc2019.org/server/validator/"
DEFAULT_OUTPUT=ROOT/"solver_checks/official_validation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def prepare(record_path,dataset,output):
    record=json.loads(record_path.read_text(encoding="utf-8"))
    name=record["instance"]
    if Path(name).name != name or not isinstance(record["seed"],int):
        raise ValueError("Unsafe instance identifier")
    source=dataset/(name+".xml")
    if sha(source)!=record["dataset_hash"]:
        raise ValueError("Dataset hash differs from saved calibration record")
    if record["score"]["hard"]!=0:
        raise ValueError("Only an internally feasible record can be exported")
    with np.load(record_path.with_suffix(".npz"),allow_pickle=False) as arrays:
        choices=arrays["choices"].copy()
    instance=load_instance(source)
    if instance.name!=name or choices.shape!=(2*len(instance.classes),):
        raise ValueError("Saved choices do not match the instance")
    for i,cls in enumerate(instance.classes):
        if not 0<=choices[i]<len(cls.times):
            raise ValueError("Time option index out of domain")
        room=choices[len(instance.classes)+i]
        if (cls.no_room and room!=-1) or (not cls.no_room and not 0<=room<len(cls.room_options)):
            raise ValueError("Room option index out of domain")
    score,assignments=evaluate(instance,choices[:len(instance.classes)],
                               choices[len(instance.classes):],keep_assignments=True)
    for key in ("hard","total","time","room","distribution","student"):
        if getattr(score,key)!=record["score"][key]:
            raise ValueError(f"Re-score drift on {key}; do not validate a different solution silently")
    validate_sectioning(instance,assignments or {})
    folder=output/f"{name}_s{record['seed']}_{record['identity']}"
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"solution.xml"
    expected_path=folder/"expected.json"
    if path.exists():
        existing=json.loads(expected_path.read_text(encoding="utf-8"))
        if existing["solution_sha256"]!=sha(path) or existing["record_sha256"]!=sha(record_path):
            raise FileExistsError("Existing validation export belongs to different content")
        return path
    spec=importlib.util.spec_from_file_location("itc_validation_export",ROOT/"NSGA-III/main.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._write_solution(path,instance,choices,score,assignments,record["seconds"],
        "NSGA-III baseline","Ho Chi Minh City University of Technology","Vietnam")
    expected={"instance":name,"seed":record["seed"],"score":asdict(score),
        "weights":instance.weights,"classes":len(instance.classes),
        "solution_sha256":sha(path),"dataset_sha256":sha(source),
        "record_sha256":sha(record_path),"choices_sha256":sha(record_path.with_suffix(".npz")),
        "record":str(record_path.resolve().relative_to(ROOT)) if record_path.is_relative_to(ROOT) else record_path.name,
        "scoring_provenance":"Re-evaluated saved choices; no new optimization; student decoder deterministic",
        "official_status":"not_run"}
    save(expected_path,expected)
    return path


def number(value):
    if isinstance(value,dict):
        value=value.get("value")
    if isinstance(value,bool):
        return None
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (TypeError,ValueError):
        return None


def compare_response(raw,expected):
    """Fail closed on unknown payload, wrong instance, incomplete or mismatched score."""
    result={"official_status":"unknown_response","official_result":None,
            "official_total":None,"internal_total":expected["score"]["total"],
            "penalty_matches":False,"official_components":{},"published":False}
    if not isinstance(raw,dict):
        return result
    obj=raw.get("obj",raw)
    if not isinstance(obj,dict):
        return result
    outcome=obj.get("result")
    result["official_result"]=outcome
    result["log"]=obj.get("log",[])
    if outcome=="NOT-VALID":
        result["official_status"]="not_valid"
        return result
    if outcome=="NOT-SOLUTION":
        result["official_status"]="custom_problem_not_competition_validation"
        return result
    if outcome!="OK":
        return result
    name=obj.get("instance")
    if isinstance(name,dict):
        name=name.get("name")
    if name!=expected["instance"]:
        result["official_status"]="instance_mismatch"
        return result
    assigned=obj.get("assignedVariables")
    if not isinstance(assigned,dict):
        return result
    if (number(assigned.get("value"))!=expected["classes"] or
            number(assigned.get("total"))!=expected["classes"]):
        result["official_status"]="incomplete_assignment"
        return result
    total=number(obj.get("totalCost"))
    if total is None:
        return result
    result["official_total"]=total
    result["official_components"]={key:obj.get(key) for key in
        ("timePenalty","roomPenalty","distributionPenalty","studentConflicts")}
    result["penalty_matches"]=total==expected["score"]["total"]
    result["official_status"]="passed" if result["penalty_matches"] else "penalty_mismatch"
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None


def submit(solution,timeout=45):
    """Use the site's actual multipart field; no retries or publication request."""
    if not 0<timeout<=120:
        raise ValueError("Validator timeout must lie in (0,120]")
    token=os.environ.get("ITC2019_TOKEN")
    if not token:
        return {"official_status":"auth_required","http_status":None,"published":False,
                "reason":"The official website requires login. Validate manually, or provide "
                         "your own session token in ITC2019_TOKEN locally; never send it in chat.",
                "network_requests":0}
    data=solution.read_bytes()
    if len(data)>20_000_000:
        raise ValueError("Validation upload exceeds the local 20 MB safety cap")
    root=ET.fromstring(data)
    if root.tag!="solution":
        raise ValueError("Submit only competition-format <solution> XML")
    boundary="itcvalidation"+uuid.uuid4().hex
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="solution"; filename="solution.xml"\r\n'
          'Content-Type: application/xml\r\n\r\n').encode()+data+f"\r\n--{boundary}--\r\n".encode()
    url=ENDPOINT+("?"+urllib.parse.urlencode({"token":token}) if token else "")
    request=urllib.request.Request(url,data=body,method="POST",headers={
        "Content-Type":"multipart/form-data; boundary="+boundary,
        "User-Agent":"ITC2019-local-validation-check/1.0"})
    started=datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.build_opener(NoRedirect).open(request,timeout=timeout) as response:
            status=response.status
            text=response.read(5_000_000).decode("utf-8",errors="replace")
    except urllib.error.HTTPError as exc:
        status=exc.code
        text=exc.read(5_000_000).decode("utf-8",errors="replace")
    except (urllib.error.URLError,TimeoutError,OSError) as exc:
        return {"official_status":"service_unavailable","http_status":None,
                "error_type":type(exc).__name__,"requested_at_utc":started,"published":False}
    try:
        raw=json.loads(text)
    except ValueError:
        raw=None
    if token:
        text=text.replace(token,"[REDACTED]")
        if raw is not None:
            raw=json.loads(json.dumps(raw).replace(token,"[REDACTED]"))
    return {"official_status":"auth_required" if status in (401,403) else
            "service_unavailable" if status>=500 else "response_received" if raw is not None else "unknown_response",
            "http_status":status,"response":raw,"non_json_excerpt":text[:500] if raw is None else None,
            "requested_at_utc":started,"endpoint":ENDPOINT,"published":False}


def expected_for(solution,expected_path):
    expected=json.loads(expected_path.read_text(encoding="utf-8"))
    if expected["score"]["hard"]!=0:
        raise ValueError("Expected score is not internally feasible")
    if sha(solution)!=expected["solution_sha256"]:
        raise ValueError("XML changed after local score/export; expected hash mismatch")
    if ET.parse(solution).getroot().get("name")!=expected["instance"]:
        raise ValueError("XML and expected-score instance differ")
    return expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    prepare_parser=sub.add_parser("prepare",help="Re-score/export saved calibration records, without GA")
    prepare_parser.add_argument("--record",type=Path,action="append",required=True)
    prepare_parser.add_argument("--dataset",type=Path,default=ROOT/"Dataset_ITC2019")
    prepare_parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    for mode in ("submit","import"):
        command=sub.add_parser(mode)
        command.add_argument("--solution",type=Path,required=True)
        command.add_argument("--expected",type=Path,help="Defaults to expected.json next to solution")
        command.add_argument("--output",type=Path,help="Defaults to official_validation.json next to solution")
        if mode=="submit":
            command.add_argument("--timeout",type=float,default=45)
        else:
            command.add_argument("--response",type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.mode=="prepare":
            for path in args.record:
                print(prepare(path.resolve(),args.dataset.resolve(),args.output.resolve()))
            return
        expected=expected_for(args.solution,args.expected or args.solution.parent/"expected.json")
        if args.mode=="submit":
            report=submit(args.solution,args.timeout)
            if report["official_status"]=="response_received":
                report.update(compare_response(report["response"],expected))
        else:
            raw=json.loads(args.response.read_text(encoding="utf-8"))
            report=compare_response(raw,expected)
            report.update(response=raw,source="manual_official_response_import")
        report.update(solution_sha256=sha(args.solution),instance=expected["instance"],
                      internal_expected=expected["score"])
        target=args.output or args.solution.parent/"official_validation.json"
        save(target,report)
        print(json.dumps({"instance":expected["instance"],"official_status":report["official_status"],
                          "report":str(target.resolve()),"published":False},ensure_ascii=False))
        if report["official_status"]!="passed":
            raise SystemExit(3)
    except (ValueError,KeyError,OSError,ET.ParseError) as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()
