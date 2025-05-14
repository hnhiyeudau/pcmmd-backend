#!/bin/bash
# Set executable permissions
chmod +x ./start.sh

# Start the application
uvicorn app.main:app --host 0.0.0.0 --port $PORT
