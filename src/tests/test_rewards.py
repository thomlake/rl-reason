import unittest

from rl_reason import reward_functions


RESPONSES_OK_STRICT = [


    """\
<think>
these are the good words
</think>
<answer>
abc efg 123
</answer>""",

    """\
<think>
these
are
the good words
</think>
<answer>
abc
efg
123
</answer>""",
]

RESPONSE_OK_SOFT = [
    """\
<think>words</think>
<answer>abc</answer>""",

    """\
<think>these are the good words</think><answer>abc efg 123</answer>""",

    """\
<think>

these are the good words

</think>


<answer>

abc efg 123

</answer>""",

    """\
<think>

these are the good words

</think>


<answer>

abc efg 123

</answer>""",
]


class TestStringMethods(unittest.TestCase):
    def test_strict_format_reward(self):
        f = reward_functions.get_strict_format_reward(scale=1)
        for s in RESPONSES_OK_STRICT:
            r, = f(completions=[s])
            self.assertEqual(r, 1, f'bad response: {repr(s)}')

        for s in RESPONSE_OK_SOFT:
            r, = f(completions=[s])
            self.assertEqual(r, 0, f'ok response: {repr(s)}')

    def test_soft_format_reward(self):
        f = reward_functions.get_soft_format_reward(scale=1)

        for s in [*RESPONSES_OK_STRICT, *RESPONSE_OK_SOFT]:
            r, = f(completions=[s])
            self.assertEqual(r, 1, f'bad response: {repr(s)}')


if __name__ == '__main__':
    unittest.main()
