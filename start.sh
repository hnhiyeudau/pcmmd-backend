#!/bin/bash
# export PYTHONPATH=src
uvicorn app.main:app --host 0.0.0.0 --port $PORT
