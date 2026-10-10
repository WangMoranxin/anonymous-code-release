from __future__ import annotations


INSTRUCTION = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request. \n\n"
    "### Instruction:\n"
    "Can you predict the next possible item that the user may expect?\n\n"
)


def _response_prompt(user_input: str) -> str:
    return f"### User Input: \n{user_input}\n\n### Response:\n"


def format_train_prompt(history: list[str]) -> str:
    joined = ", ".join(history)
    user_input = (
        f"The user has interacted with items {joined} in chronological order. "
        "Can you predict the next possible item that the user may expect?"
    )
    return _response_prompt(user_input)


def format_test_prompt(history: list[str]) -> str:
    joined = ", ".join(history)
    user_input = (
        "Can you predict the next possible item the user may expect, given the "
        f"following chronological interaction history: {joined}"
    )
    return INSTRUCTION + _response_prompt(user_input)


def format_prompt(history: list[str], split: str) -> str:
    return format_test_prompt(history) if split == "test" else format_train_prompt(history)
