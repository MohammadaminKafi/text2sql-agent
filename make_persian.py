# -*- coding: utf-8 -*-
import json, glob
from translation_map import translations

for path in sorted(glob.glob('datasets/dataset_AdventureWorks2022/*/prompt*.json')):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    translated = {k: translations.get(v, v) for k,v in data.items()}
    new_path = path.replace('prompt', 'persian_prompt')
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
