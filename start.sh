#!/bin/bash

set -e

mkdir -p model

echo "Verificando model.safetensors..."

if [ ! -f "model/model.safetensors" ]; then
  echo "Baixando model.safetensors do Google Drive..."
  gdown "1AjSGD5_lGQU03M7ap7_kxC64sxGsMQce" -O model/model.safetensors
else
  echo "model.safetensors já existe."
fi

echo "Verificando tokenizer..."

if [ ! -f "model/zyluncpt_tokenizer.json" ]; then
  echo "ERRO: model/zyluncpt_tokenizer.json não encontrado."
  echo "Coloque o tokenizer no repo ou adicione outro link para baixar ele também."
  exit 1
fi

echo "Iniciando API..."
uvicorn main:app --host 0.0.0.0 --port $PORT
