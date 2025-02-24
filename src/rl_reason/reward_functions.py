import re


def _extract_completion_text(obj: str | dict) -> str:
    if isinstance(obj, str):
        return obj

    return obj[0]['content']


def get_strict_format_reward(scale: float = 0.5):
    pattern = re.compile(
        r'^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$',
        flags=re.DOTALL,
    )

    def strict_format_reward(completions, **kwargs):
        completions = [_extract_completion_text(c) for c in completions]
        matches = [pattern.match(c) for c in completions]
        return [scale if match else 0.0 for match in matches]

    return strict_format_reward


def get_soft_format_reward(scale: float = 0.5, add_think: bool = True):
    pattern = re.compile(
        r'^<think>.*?</think>\s*<answer>.*?</answer>$',
        flags=re.DOTALL | re.MULTILINE,
    )

    def soft_format_reward(completions, **kwargs):
        completions = [_extract_completion_text(c) for c in completions]
        if add_think:
            completions = ["<think>" + c for c in completions]

        matches = [pattern.match(c) for c in completions]
        return [scale if match else 0.0 for match in matches]

    return soft_format_reward


def get_countdown_reward(scale: float = 1.0):
    answer_pattern = re.compile(
        r'<answer>(.*?)</answer>',
        re.DOTALL,
    )
    digit_pattern = re.compile(r'\d+')
    eqn_pattern = re.compile(r'^[\d+\-*/().\s]+$')

    def countdown_reward(completions, answer, nums, **kwargs):
        rewards = []
        for completion, answer_true, numbers in zip(completions, answer, nums):
            try:
                # Check if the format is correct
                match = answer_pattern.search(completion)
                if match is None:
                    rewards.append(0.0)
                    continue

                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in digit_pattern.findall(equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue

                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                if not eqn_pattern.match(equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builtins__": None}, {})

                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(answer_true)) < 1e-5:
                    rewards.append(scale)
                else:
                    rewards.append(0.0)

            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)

        return rewards

    return countdown_reward


REWARD_FUNCTION_REGISTRY = {
    'strict_format': get_strict_format_reward,
    'soft_format': get_soft_format_reward,
    'countdown': get_countdown_reward,
}


def create_reward_functions(reward_config: dict[str, dict]):
    return [
        REWARD_FUNCTION_REGISTRY[name](**kwargs)
        for name, kwargs in reward_config.items()
    ]
