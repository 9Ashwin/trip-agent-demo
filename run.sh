#!/bin/bash
# 一键启动脚本
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "首次运行，正在创建虚拟环境并安装依赖…"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "已生成 .env（当前为空 → 会以演示模式启动）"
fi

echo "启动中，浏览器打开 http://127.0.0.1:5050"
.venv/bin/python app.py
