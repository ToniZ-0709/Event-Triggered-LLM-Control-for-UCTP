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
        monkeypatch.setattr(llm,"create_decision_client",lambda *a,**kw:
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


@pytest.mark.parametrize("pipeline_dir,model_profile,expect_provider", [
    ("Single-Agent Controller", "gemini_hydrogen", "gemini"),
    ("Multi-Agent Controller", "cloudflare_llama", "cloudflare"),
    ("Single-Agent Controller", "cloudflare_mistral", "cloudflare"),
])
def test_controller_cli_hashes_all_provider_sources_and_audits_cleanly(
    tmp_path, monkeypatch, pipeline_dir, model_profile, expect_provider
):
    root = BASE.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from providers import cloudflare_pool, gemini_pool
    from providers.settings import create_decision_client
    from tools.audit_runs import audit_attempt
    import httpx

    dataset = tmp_path / "data"
    dataset.mkdir()
    option = '<room id="1" penalty="0"/><room id="2" penalty="0"/>'
    times = '<time days="1" weeks="1" start="0" length="1" penalty="0"/><time days="1" weeks="1" start="1" length="1" penalty="0"/>'
    (dataset / "tiny.xml").write_text(
        '<problem name="tiny" nrDays="1" nrWeeks="1" slotsPerDay="4">'
        '<optimization time="1" room="1" distribution="1" student="1"/>'
        '<rooms><room id="1"/><room id="2"/></rooms><courses>' +
        ''.join(f'<course id="{i}"><config id="{i}"><subpart id="{i}">'
                f'<class id="{i}" limit="10">{option}{times}</class>'
                '</subpart></config></course>' for i in (1, 2)) +
        '</courses><distributions/><students/></problem>',
        encoding="utf-8",
    )

    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY_001=offline-test-gemini-key\n"
        "CLOUDFLARE_ACCOUNT_ID_001=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "CLOUDFLARE_API_TOKEN_001=offline-test-cf-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gemini_pool, "ENV_PATH", env_file)
    monkeypatch.setattr(cloudflare_pool, "ENV_PATH", env_file)
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")

    controller_main = root / pipeline_dir / "main.py"
    monkeypatch.syspath_prepend(str(controller_main.parent))
    monkeypatch.delitem(sys.modules, "controller", raising=False)
    spec = importlib.util.spec_from_file_location(
        f"tested_controller_cli_{pipeline_dir.split()[0].lower()}", controller_main
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def make_client(settings, *, thinking_level=None, free_tier_confirmed=False,
                    timeout_seconds=None, max_output_tokens=None):
        profile = {**settings, "min_request_interval_seconds": 0}
        if profile["provider"] == "gemini":
            pool = gemini_pool.GeminiPool(
                profile["model"], profile["quota"], env_path=env_file,
                usage_path=tmp_path / "g_usage.json", legacy_path=tmp_path / "missing.json",
            )
            answer = json.dumps({
                "solution_id": None, "region_id": None, "boundary_id": None,
                "operator_id": "no_op", "execution_budget": 0, "next_review_interval": 3,
            })
            transport = httpx.MockTransport(lambda req: httpx.Response(200, json={
                "modelVersion": profile["model"],
                "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": answer}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            }))
        else:
            pool = cloudflare_pool.CloudflarePool(env_path=env_file, usage_path=tmp_path / "cf_usage.json")
            def cf_handler(req):
                if pipeline_dir.startswith("Single"):
                    payload = {"solution_id": None, "region_id": None, "boundary_id": None,
                               "operator_id": "no_op", "execution_budget": 0, "next_review_interval": 3}
                else:
                    payload = {"solution_id": None, "region_id": None, "boundary_id": None,
                               "relevant_constraint_ids": []}
                body_ans = json.dumps(payload) if profile["model"].endswith("instruct") and "mistral" in profile["model"] else payload
                return httpx.Response(200, json={"success": True, "result": {
                    "response": body_ans,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }})
            transport = httpx.MockTransport(cf_handler)
        return create_decision_client(
            profile, thinking_level=thinking_level, free_tier_confirmed=free_tier_confirmed,
            timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens,
            pool=pool, transport=transport,
        )

    monkeypatch.setattr(llm, "create_decision_client", make_client)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        str(controller_main), "--dataset", str(dataset), "--instance", "tiny",
        "--output", str(output), "--model", model_profile, "--confirm-free-tier",
        "--population", "4", "--partitions", "1", "--generations", "6",
        "--warmup-generations", "2", "--review-interval", "2", "--min-review-interval", "1",
        "--initial-repair-steps", "0", "--offspring-repair-steps", "0", "--quiet",
    ])
    module.main()
    run_dir, = list(output.glob("run_*"))
    manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    settings = manifest["settings"]
    assert settings["controller"]["provider"] == expect_provider
    assert settings["llm"]["provider"] == expect_provider
    src_hashes = {k.replace("\\", "/") for k in settings["controller_sources_sha256"]}
    for expected_src in (
        "providers/settings.py", "providers/decision_client.py", "providers/credential_pool.py",
        "providers/gemini.py", "providers/gemini_pool.py",
        "providers/cloudflare.py", "providers/cloudflare_pool.py",
    ):
        assert expected_src in src_hashes
    method = "single" if pipeline_dir.startswith("Single") else "multi"
    audit = audit_attempt({"instance": "tiny", "seed": 17, "method": method}, run_dir)
    assert audit["passed"], audit["errors"]
