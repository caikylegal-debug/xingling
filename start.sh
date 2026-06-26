#!/usr/bin/env bash

mkdir -p zyluncpt_0_1_instruct_final

echo "Baixando arquivos grandes do Google Drive..."

pip install gdown

if [ ! -f zyluncpt_0_1_instruct_final/model.safetensors ]; then
  gdown "https://drive.google.com/uc?id=1AjSGD5_lGQU03M7ap7_kxC64sxGsMQce" -O zyluncpt_0_1_instruct_final/model.safetensors
fi

if [ ! -f zyluncpt_0_1_instruct_final/train_state.pt ]; then
  gdown "https://drive.google.com/uc?id=1hnDuQ3hZQ4KCCx4UpjT1XU4matj1oS6E" -O zyluncpt_0_1_instruct_final/train_state.pt
fi

echo "Arquivos baixados. Iniciando servidor..."

python app.py
