#!/bin/bash

ACTION=$1
MODE=${2:-sync}
PORT_SYNC=50051
PORT_ASYNC=50052
OLLAMA_HOST="http://localhost:11434"
DEVICE="cuda"
LOG_DIR="./logs"
PYTHON_BIN="python3.12"

mkdir -p "$LOG_DIR"

update_code() {
  echo "[UPDATE] Pulling latest code..."
  git pull --ff-only || { echo "[ERROR] Git pull failed"; exit 1; }

  echo "[INSTALL] Reinstalling olol package..."
  $PYTHON_BIN -m pip install -e . || { echo "[ERROR] pip install failed"; exit 1; }
}

start() {
  update_code

  if [ "$MODE" == "sync" ]; then
    echo "[START] Sync server"
    nohup olol server --host 0.0.0.0 --port $PORT_SYNC --ollama-host $OLLAMA_HOST > $LOG_DIR/server.log 2>&1 &
  elif [ "$MODE" == "async" ]; then
    echo "[START] Async RPC server"
    nohup olol rpc-server --host 0.0.0.0 --port $PORT_ASYNC --device $DEVICE --flash-attention --context-window 16384 --quantize q5_0 > $LOG_DIR/rpc.log 2>&1 &
  fi
}

stop() {
  echo "[STOP] Killing olol server processes"
  pkill -f "olol server"
  pkill -f "olol rpc-server"
}

restart() {
  stop
  sleep 1
  start
}

status() {
  echo "[STATUS]"
  pgrep -a -f "olol server"
  pgrep -a -f "olol rpc-server"
  ss -tuln | grep 500
}

case "$ACTION" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  *) echo "Usage: $0 {start|stop|restart|status} [sync|async]" ;;
esac