#!/usr/bin/env python3
import sys

input_file = 'experiments/educational_scripts/education_state_mapper_v2.py.bak'
output_file = 'experiments/educational_scripts/education_state_mapper_v2.py'

with open(input_file, 'rb') as f:
    content = f.read()

# 移除损坏的 UTF-8 字符 (U+FFFD = EF BF BD)
content = content.replace(b'\xef\xbf\xbd', b'')

# 转换为字符串
text = content.decode('utf-8', errors='replace')

# 修复常见乱码 - 使用原始字节
replacements = [
    (b'\xe6\x98\xa0'.decode('utf-8'), '映射'),  # 映
    (b'\xe5\x88\x9b'.decode('utf-8'), '创'),   # 创
    (b'\xe5\xa4\x9a\xe5\xb7\xb4'.decode('utf-8'), '多巴'),  # 多巴
]

for old, new in replacements:
    text = text.replace(old, new)

# 额外修复
text = text.replace('??', '')
text = text.replace('？', '')
text = text.replace('：', ':')
text = text.replace('？', '?')
text = text.replace('（', '(')
text = text.replace('）', ')')
text = text.replace('，', ',')
text = text.replace('。', '.')
text = text.replace('→', '-')

text = text.replace('映射射', '映射')
text = text.replace('学生状？', '学生状态')
text = text.replace('神经调质状？', '神经调质状态')
text = text.replace('V2 映射器？', 'V2 映射器')

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Fixed! Output: {output_file}')
