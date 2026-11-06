#!/bin/zsh
cd "$(dirname "$0")"
echo "Starte HealthPolicy Simulator Österreich auf http://127.0.0.1:8501 ..."
(sleep 1.2 && open "http://127.0.0.1:8501") &
python3 server.py 8501
