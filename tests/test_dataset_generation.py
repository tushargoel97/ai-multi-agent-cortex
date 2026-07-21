from trainer import generate_dataset


def test_soc_buying_prompts_are_not_labeled_desktop_cpus():
    examples = generate_dataset._buying_examples(
        [
            {
                "_group": "cpus",
                "name": "Mobile Chip",
                "category": "SoC",
                "release_year": 2021,
                "best_for": ["phones"],
            }
        ]
    )

    questions = [example["messages"][0]["content"] for example in examples]
    assert questions and all("mobile SoC" in question for question in questions)


def test_generate_returns_description_for_selected_training_domains(monkeypatch):
    monkeypatch.setattr(
        generate_dataset,
        "available_domains",
        lambda: [
            {
                "name": "legal",
                "description": "Legal documents and obligations.",
                "subdomains": [
                    {
                        "name": "contracts",
                        "label": "Contracts",
                        "description": "Commercial contracts, clauses, and termination rights.",
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(generate_dataset, "_selected_keys", lambda *_: {"legal/contracts"})
    monkeypatch.setattr(generate_dataset, "_hardware_groups", lambda: [])
    monkeypatch.setattr(generate_dataset, "_subdomain_dirs", lambda: [])
    monkeypatch.setattr(generate_dataset, "_identity_examples", lambda _: [])
    monkeypatch.setattr(
        generate_dataset,
        "write_splits",
        lambda _: {"train_count": 1, "valid_count": 1},
    )

    result = generate_dataset.generate(subdomains=["legal/contracts"])

    assert result["description"] == (
        "Fine-tuned specialist for Contracts in legal: Commercial contracts, "
        "clauses, and termination rights."
    )
