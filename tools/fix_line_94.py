# -*- coding: utf-8 -*-
with open(r'd:\python_projects\NCT\experiments\NCT_value_experiments_design\exp2_interpretability_v1_final.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Print line 94 (index 93)
print("Line 94:")
print(repr(lines[93]))

# Fix the line
lines[93] = '                    mask = (x - 14)**2 + **(y - 14)2 <= radius**2\n'

with open(r'd:\python_projects\NCT\experiments\NCT_value_experiments_design\exp2_interpretability_v1_final.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\nFixed!")
print("New line 94:")
print(repr(lines[93]))
