#!/bin/bash

HOST="0.0.0.0"
PORT=8000
# Définir les serveurs individuellement
SERVER_1="10.0.9.245:50051"
SERVER_2="10.0.9.248:50051"
SERVER_3="10.0.9.249:50051"
# Liste de serveurs séparés pour les commandes Python
SERVER_LIST="$SERVER_1,$SERVER_2,$SERVER_3"
# Liste de serveurs sous forme d'array pour les tests
SERVERS=($SERVER_1 $SERVER_2 $SERVER_3)

# Définir les serveurs RPC individuellement
RPC_1="10.0.9.245:50052"
RPC_2="10.0.9.248:50052"
RPC_3="10.0.9.249:50052"
# Liste de serveurs RPC séparés pour les commandes Python
RPC_SERVER_LIST="$RPC_1,$RPC_2,$RPC_3"

LOG_DIR="./logs"
PYTHON_BIN="python3.12"

mkdir -p "$LOG_DIR"

# Fonction pour mettre à jour le code depuis git
update_code() {
  echo "[UPDATE] Pulling latest code..."
  git pull --ff-only || { echo "[ERROR] Git pull failed"; exit 1; }

  echo "[INSTALL] Réinstallation en mode développement..."
  $PYTHON_BIN -m pip install -e . || { echo "[ERROR] pip install failed"; exit 1; }
}

# Lancer le proxy en mode direct (exécute Python directement sans passer par le point d'entrée olol-proxy)
run_direct() {
  echo "[DIRECT] Lancer le proxy directement avec Python pour voir les erreurs en direct"
  $PYTHON_BIN -c "from olol.proxy import run_proxy; run_proxy(host='$HOST', port=$PORT, server_addresses=['$SERVER_1', '$SERVER_2', '$SERVER_3'], enable_distributed=True, rpc_servers=['$RPC_1', '$RPC_2', '$RPC_3'], debug=True, verbose=True)"
}

# Lancer le proxy avec l'entrée olol-proxy en mode debug
run_debug() {
  echo "[DEBUG] Lancer le proxy en mode debug"
  PYTHONPATH=$(pwd) FLASK_APP=olol.proxy FLASK_DEBUG=1 $PYTHON_BIN -m olol.proxy --host $HOST --port $PORT --servers "$SERVER_1,$SERVER_2,$SERVER_3" --rpc-servers "$RPC_1,$RPC_2,$RPC_3" --distributed --verbose --debug
}

# Lancer le proxy avec un redirecteur de sortie pour capturer tous les logs
run_logged() {
  echo "[LOGGED] Lancer le proxy avec logs détaillés"
  PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 $PYTHON_BIN -m olol.proxy --host $HOST --port $PORT --servers "$SERVER_1,$SERVER_2,$SERVER_3" --rpc-servers "$RPC_1,$RPC_2,$RPC_3" --distributed --verbose --debug > $LOG_DIR/proxy-debug.log 2>&1
}

# Tester chaque nœud individuellement pour vérifier la connectivité
test_nodes() {
  echo "[TEST] Vérification de la connectivité avec chaque nœud..."
  
  for node in "${SERVERS[@]}"; do
    echo "Testing connectivity to $node"
    $PYTHON_BIN -c "from olol.sync.client import OllamaClient; client = OllamaClient(host='${node%:*}', port=${node#*:}); print('Health check:', client.check_health()); models = client.list_models(); print('Models:', [m.name for m in models.models] if hasattr(models, 'models') else 'Error: ' + str(models))"
  done
}

# Exécuter des commandes curl de test basiques
run_tests() {
  echo "[TEST] Test de l'API status"
  curl -m 5 http://$HOST:$PORT/api/status
  
  echo -e "\n\n[TEST] Test de l'API models"
  curl -m 5 http://$HOST:$PORT/api/models
  
  echo -e "\n\n[TEST] Test de l'API generate avec timeout"
  curl -m 10 -X POST http://$HOST:$PORT/api/generate \
    -H "Content-Type: application/json" \
    -d '{"model": "mistral", "prompt": "Hello, how are you?", "stream": false}'
}

case "$1" in
  update)
    update_code
    ;;
  direct)
    run_direct
    ;;
  debug)
    run_debug
    ;;
  logged)
    run_logged
    ;;
  test-nodes)
    test_nodes
    ;;
  test-api)
    run_tests
    ;;
  *)
    echo "Usage: $0 {update|direct|debug|logged|test-nodes|test-api}"
    echo ""
    echo "  update      - Met à jour le code depuis git et réinstalle"
    echo "  direct      - Lance le proxy directement avec Python pour voir les erreurs en direct"
    echo "  debug       - Lance le proxy en mode debug avec Flask"
    echo "  logged      - Lance le proxy avec logs détaillés dans logs/proxy-debug.log"
    echo "  test-nodes  - Teste la connectivité avec chaque nœud"
    echo "  test-api    - Exécute des tests basiques d'API"
    ;;
esac