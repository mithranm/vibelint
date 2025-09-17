#!/usr/bin/env python3
"""
Systematic emoji removal script for vibelint.
Replaces emojis with descriptive text alternatives.
"""

import re
from pathlib import Path

# Emoji mapping for common ones found in vibelint
EMOJI_REPLACEMENTS = {
    # Search/analysis
    "🔍": "[SEARCH]",
    "📊": "[STATS]",
    "📈": "[CHART]",
    "📉": "[GRAPH]",

    # Status indicators
    "✅": "[PASS]",
    "❌": "[FAIL]",
    "✓": "[OK]",
    "✗": "[ERROR]",
    "⚠️": "[WARNING]",
    "🚨": "[ALERT]",

    # Actions
    "🔄": "[REFRESH]",
    "💾": "[SAVE]",
    "📝": "[EDIT]",
    "📋": "[REPORT]",
    "📁": "[FOLDER]",
    "🔧": "[TOOL]",

    # Messages
    "💡": "[TIP]",
    "🎉": "[SUCCESS]",
    "⏱️": "[TIMER]",
    "🏗️": "[BUILD]",
    "🏛️": "[ARCH]",
    "📄": "[DOC]",
    "🚀": "[ROCKET]",
    "📤": "[EXPORT]",
    "🔗": "[LINK]",
}

def remove_emojis_from_file(file_path: Path) -> bool:
    """Remove emojis from a single file. Returns True if changes were made."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Apply specific replacements first
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)

        # Comprehensive emoji pattern for any remaining emojis
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F"  # Emoticons
            r"\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
            r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
            r"\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols
            r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            r"\U00002600-\U000026FF"  # Miscellaneous Symbols
            r"\U00002700-\U000027BF"  # Dingbats
            r"\U0000FE00-\U0000FE0F"  # Variation Selectors
            r"]+"
        )

        # Remove any remaining emojis
        content = emoji_pattern.sub("[EMOJI]", content)

        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"Fixed emojis in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Process all Python files in src/vibelint directory."""
    src_dir = Path("src/vibelint")
    if not src_dir.exists():
        print("src/vibelint directory not found")
        return

    python_files = list(src_dir.rglob("*.py"))
    fixed_count = 0

    for file_path in python_files:
        if remove_emojis_from_file(file_path):
            fixed_count += 1

    print(f"\nFixed emojis in {fixed_count} files out of {len(python_files)} Python files")

if __name__ == "__main__":
    main()