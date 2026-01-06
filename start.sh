#!/bin/bash
set -e

# 1. Минимальная установка (тихо, быстро)
apt-get update -qq && apt-get install -y -qq git

# 2. Клонируем ТОЛЬКО исходники (экономим время и трафик)
git clone --depth=1 --filter=blob:none --quiet https://github.com/HynekPetrak/javascript-malware-collection.git
git clone --depth=1 --filter=blob:none --quiet https://github.com/highlightjs/highlight.js.git

# 3. Установка (без лишнего)
pip3 install --no-cache-dir -q torch transformers scikit-learn

# 4. Запуск: обучаем на malware + ТОЛЬКО src/ из highlight.js
python3 train_gelb.py \
  --train \
  --train_data "javascript-malware-collection,highlight.js/src" \
  --epochs 5 \
  --batch_size 1
