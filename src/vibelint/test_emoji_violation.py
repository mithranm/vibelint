"""
Test file with emoji violations to test Claude Code hooks.

This file contains emojis 🚀 and should trigger self-validation warnings.
"""


def hello_world():
    """A function with emoji in docstring 😀 🔥"""
    print("Hello World! 🌍 🚀")  # This emoji should be caught
    return "Success! ✅ 🎉"  # Another emoji violation


# Variable with emoji name
rocket_emoji = "🚀"
celebration = "🎊 🎈 🎁 🎂 🍰"  # Testing consolidated hook


class EmojiClass:
    """Class with emoji 🎉"""

    def method_with_emoji(self):
        """Method with emoji 🔥"""
        return "Done! 👍"
