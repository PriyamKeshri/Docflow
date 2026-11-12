"""Static checks on the n8n workflow exports (they can't be unit-tested inside n8n here)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / "n8n" / "workflows").glob("*.json"))


def test_expected_workflows_exist() -> None:
    assert {p.name for p in WORKFLOWS} == {
        "docflow-intake.json",
        "docflow-notify.json",
        "docflow-nightly-batch.json",
    }


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_is_well_formed(path: Path) -> None:
    workflow = json.loads(path.read_text())
    names = [node["name"] for node in workflow["nodes"]]
    assert len(names) == len(set(names)), "node names must be unique"
    assert len({node["id"] for node in workflow["nodes"]}) == len(names)
    for source, outputs in workflow["connections"].items():
        assert source in names
        for branch in outputs["main"]:
            for edge in branch:
                assert edge["node"] in names, f"{source} -> {edge['node']}"
    assert workflow["active"] is False  # imported inactive; activate after adding credentials


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_secrets_are_hard_coded(path: Path) -> None:
    text = path.read_text()
    assert "hooks.slack.com" not in text
    for node in json.loads(text)["nodes"]:
        assert "credentials" not in node  # chosen in the n8n UI after import
        for header in node["parameters"].get("headerParameters", {}).get("parameters", []):
            assert header["value"].startswith("={{ $env.")


def test_webhook_paths_match_the_api_defaults() -> None:
    paths = {
        node["parameters"]["path"]
        for path in WORKFLOWS
        for node in json.loads(path.read_text())["nodes"]
        if node["type"] == "n8n-nodes-base.webhook"
    }
    assert paths == {"docflow-intake", "docflow-notify"}
    env_example = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert "/webhook/docflow-notify" in env_example
