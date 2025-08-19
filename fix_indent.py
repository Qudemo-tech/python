#!/usr/bin/env python3
"""Fix indentation issue in enhanced_qa.py"""

with open('enhanced_qa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic line
content = content.replace('                        continue', '                    continue')

with open('enhanced_qa.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed indentation issue!")
