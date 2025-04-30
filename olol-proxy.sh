#!/bin/bash

ACTION=$1
PORT=8000
SERVERS="10.0.9.245:50051,10.0.9.248:50051,10.0.9.249:50051"
RPC_SERVERS="10.0.9.245:50052,10.0.9.248:50052,10.0.9.249:50052"
LOG="./proxy.log"
DISTRIBUTED=true
DISCOVERY=true
INTERFACE=""
PYTHON_BIN="python3.12"

update_code() {
  echo "[UPDATE] Pulling latest code..."
  git pull --ff-only || { echo "[ERROR] Git pull failed"; exit 1; }

  echo "[INSTALL] Reinstalling osync package..."
  $PYTHON_BIN -m pip install -e . || { echo "[ERROR] pip install failed"; exit 1; }
}

start() {
  update_code

  echo "[START] Proxy frontend"
  nohup osync proxy \
    --host 0.0.0.0 \
    --port $PORT \
    --servers "$SERVERS" \
    --rpc-servers "$RPC_SERVERS" \
    ${DISTRIBUTED:+--distributed} \
    ${DISCOVERY:+--discovery} \
    ${INTERFACE:+--interface $INTERFACE} > $LOG 2>&1 &
}

stop() {
  echo "[STOP] Killing osync proxy"
  pkill -f "osync proxy"
}

restart() {
  stop
  sleep 1
  start
}

status() {
  echo "[STATUS]"
  pgrep -a -f "osync proxy"
  ss -tuln | grep $PORT
  tail -n 20 $LOG
}

case "$ACTION" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  *) echo "Usage: $0 {start|stop|restart|status}" ;;
esac