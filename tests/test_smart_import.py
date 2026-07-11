import json

import pytest

from trainer.app import domains, research


def test_existing_subdomain_reextracts_with_its_full_schema(monkeypatch):
    fields = [
        {"key": key, "label": key.title()}
        for key in ("cpu", "gpu", "memory", "display")
    ]
    replies = iter(
        [
            {
                "domain": "hardware",
                "subdomain": "cpus",
                "render": "spec_table",
                "fields": [{"key": "gpu", "label": "GPU"}],
                "entities": [{"name": "Chip", "gpu": "GPU"}],
            },
            {
                "entities": [
                    {
                        "name": "Chip",
                        "cpu": "CPU",
                        "gpu": "GPU",
                        "memory": "RAM",
                        "display": "4K",
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(research, "_import_gather_text", lambda *_: "source")
    monkeypatch.setattr(research, "_existing_taxonomy", lambda: "hardware: cpus")
    monkeypatch.setattr(research, "_subdomain_exists", lambda *_: True)
    monkeypatch.setattr(research, "_chat", lambda _: json.dumps(next(replies)))
    monkeypatch.setattr(
        domains,
        "get_subdomain",
        lambda *_: {"fields": fields, "render": "spec_table"},
    )
    monkeypatch.setattr(domains, "_hw_groups", lambda: [])

    proposal = research.propose(["source"], "auto")

    assert proposal["fields"] == fields
    assert proposal["entities"][0]["cpu"] == "CPU"
    assert proposal["entities"][0]["memory"] == "RAM"


def test_sparse_hardware_rows_are_enriched_or_rejected(monkeypatch):
    monkeypatch.setattr(
        research,
        "research_product",
        lambda *_args, **_kwargs: {
            "name": "Chip",
            "cpu": "CPU",
            "gpu": "GPU",
            "memory": "RAM",
            "display": "4K",
            "aliases": ["Full"],
        },
    )

    rows = research._enrich_hardware_entities(
        [{"name": "Chip", "gpu": "Source GPU", "aliases": ["Short"]}],
        lambda _: None,
    )

    assert rows[0]["gpu"] == "Source GPU"
    assert rows[0]["cpu"] == "CPU"
    assert rows[0]["aliases"] == ["Full", "Short"]

    monkeypatch.setattr(research, "research_product", lambda *_args, **_kwargs: None)
    assert (
        research._enrich_hardware_entities(
            [{"name": "Thin", "gpu": "GPU"}], lambda _: None
        )
        == []
    )


def test_hardware_apply_merges_instead_of_erasing_known_fields(monkeypatch):
    captured = {}
    monkeypatch.setattr(domains, "_hw_groups", lambda: ["cpus"])
    monkeypatch.setattr(
        domains,
        "get_subdomain",
        lambda *_: {
            "entities": [
                {"name": "Chip", "cpu": "Known CPU", "memory": "Known RAM"}
            ]
        },
    )
    monkeypatch.setattr(
        domains,
        "set_entities",
        lambda _domain, _group, rows: captured.setdefault("rows", rows),
    )

    research._apply_hardware_import(
        "cpus",
        [
            {
                "name": "Chip",
                "release_year": 2025,
                "gpu": "New GPU",
                "storage": "1 TB",
                "display": "4K",
                "key_features": ["Feature"],
            }
        ],
    )

    assert captured["rows"] == [
        {
            "name": "Chip",
            "cpu": "Known CPU",
            "memory": "Known RAM",
            "release_year": 2025,
            "gpu": "New GPU",
            "storage": "1 TB",
            "display": "4K",
            "key_features": ["Feature"],
        }
    ]


def test_hardware_apply_rejects_sparse_rows(monkeypatch):
    monkeypatch.setattr(domains, "_hw_groups", lambda: ["cpus"])
    monkeypatch.setattr(
        domains,
        "get_subdomain",
        lambda *_: {"entities": [{"name": "Chip", "cpu": "Known CPU"}]},
    )
    monkeypatch.setattr(
        domains,
        "set_entities",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not write")),
    )

    with pytest.raises(ValueError, match="incomplete"):
        research._apply_hardware_import("cpus", [{"name": "Chip", "gpu": "GPU"}])


def test_generic_import_merges_existing_fields(monkeypatch):
    captured = {}
    monkeypatch.setattr(domains, "_pack_learned_path", lambda *_: "learned.yaml")
    monkeypatch.setattr(
        domains,
        "_read_yaml",
        lambda *_: {"learned": [{"name": "Item", "summary": "Known"}]},
    )
    monkeypatch.setattr(
        domains,
        "_write_yaml",
        lambda _path, data: captured.update(data),
    )

    rows = domains.add_learned_entities("domain", "subdomain", [{"name": "Item", "year": 2025}])

    assert rows == [{"name": "Item", "summary": "Known", "year": 2025, "exists": True}]
    assert captured == {"learned": rows}
