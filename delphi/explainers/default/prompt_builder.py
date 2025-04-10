from .prompts import example, system, system_single_token


def build_examples(
    **kwargs,
):
    examples = []

    for i in range(1, 4):
        prompt, response = example(i, **kwargs)

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        examples.extend(messages)

    return examples


def build_prompt(
    examples: str,
    activations: bool = False,
    cot: bool = False,
) -> list[dict]:
    messages = system(
        cot=cot,
    )

    # Creazione dei 3 esempi da dare all'explainer
    few_shot_examples = build_examples(
        activations=activations,
        cot=cot,
    )
    
    # Aggiunta dei 3 esempi al prompt (che specifica cosa deve fare l'explainer)
    messages.extend(few_shot_examples)

    user_start = f"\n{examples}\n" # "examples" sono gli esempi da spiegare (è il parametro passato) 

    # Aggiungo gli esempi da spiegare al messaggio sa passare all'explainer
    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages


def build_single_token_prompt(
    examples,
):
    messages = system_single_token()

    user_start = f"WORDS: {examples}"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages
