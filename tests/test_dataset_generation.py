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
