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

# Exécuter un test complet de tous les endpoints API
test_all_endpoints() {
  echo "==================================================================="
  echo "                   TEST COMPLET DES ENDPOINTS API                   "
  echo "==================================================================="
  
  # 1. GET /api/status - État du serveur
  echo -e "\n[TEST 1/8] GET /api/status - État du serveur"
  curl -s -m 5 http://$HOST:$PORT/api/status | jq . || echo "Erreur: /api/status ne répond pas"
  
  # 2. GET /api/models - Liste des modèles disponibles
  echo -e "\n[TEST 2/8] GET /api/models - Liste des modèles disponibles"
  curl -s -m 5 http://$HOST:$PORT/api/models | jq . || echo "Erreur: /api/models ne répond pas"
  
  # 3. GET /api/servers - Liste des serveurs du cluster
  echo -e "\n[TEST 3/8] GET /api/servers - Liste des serveurs du cluster"
  curl -s -m 5 http://$HOST:$PORT/api/servers | jq . || echo "Erreur: /api/servers ne répond pas"
  
  # 4. POST /api/generate - Génération de texte
  echo -e "\n[TEST 4/8] POST /api/generate - Génération de texte"
  curl -s -m 10 -X POST http://$HOST:$PORT/api/generate \
    -H "Content-Type: application/json" \
    -d '{"model": "mistral", "prompt": "Expliquez brièvement ce quest un LLM.", "stream": false}' | jq . || echo "Erreur: /api/generate ne répond pas"
    
  # 5. POST /api/chat - Chat avec modèle
  echo -e "\n[TEST 5/8] POST /api/chat - Chat avec modèle"
  curl -s -m 10 -X POST http://$HOST:$PORT/api/chat \
    -H "Content-Type: application/json" \
    -d '{"model": "mistral", "messages": [{"role": "user", "content": "Bonjour, comment ça va?"}], "stream": false}' | jq . || echo "Erreur: /api/chat ne répond pas"
    
  # 6. POST /api/embeddings - Obtention d'embeddings
  echo -e "\n[TEST 6/8] POST /api/embeddings - Obtention d'embeddings"
  curl -s -m 10 -X POST http://$HOST:$PORT/api/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model": "mistral", "prompt": "Ceci est un test d'\''embedding"}' | jq . || echo "Erreur: /api/embeddings ne répond pas"
    
  # 7. GET /api/models/:model/context - Informations sur le contexte d'un modèle
  echo -e "\n[TEST 7/8] GET /api/models/:model/context - Informations sur le contexte"
  curl -s -m 5 http://$HOST:$PORT/api/models/mistral/context | jq . || echo "Erreur: /api/models/:model/context ne répond pas"
    
  # 8. POST /api/transfer - Test de transfert de modèle
  echo -e "\n[TEST 8/8] POST /api/transfer - Test de transfert de modèle"
  # Récupérer d'abord la liste des serveurs
  SERVERS_JSON=$(curl -s -m 5 http://$HOST:$PORT/api/servers)
  
  # Extraire les deux premiers serveurs s'ils existent
  if [ ! -z "$SERVERS_JSON" ]; then
    # Utiliser jq pour extraire les clés du dictionnaire 'servers' (ce sont les adresses des serveurs)
    SERVER_ADDRESSES=$(echo $SERVERS_JSON | jq -r '.servers | keys | .[0:2][]' 2>/dev/null)
    
    # Convertir en tableau
    readarray -t SERVER_ARRAY <<< "$SERVER_ADDRESSES"
    
    if [ ${#SERVER_ARRAY[@]} -ge 2 ]; then
      SOURCE_SERVER="${SERVER_ARRAY[0]}"
      TARGET_SERVER="${SERVER_ARRAY[1]}"
      
      echo "  Tentative de transfert du modèle 'mistral' de $SOURCE_SERVER vers $TARGET_SERVER"
      curl -s -m 10 -X POST http://$HOST:$PORT/api/transfer \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"mistral\", \"source\": \"$SOURCE_SERVER\", \"target\": \"$TARGET_SERVER\"}" | jq . || echo "Erreur: /api/transfer ne répond pas"
    else
      echo "  Pas assez de serveurs disponibles pour tester le transfert"
    fi
  else
    echo "  Impossible de récupérer la liste des serveurs"
  fi
  
  echo -e "\n==================================================================="
  echo "                      TEST DES ENDPOINTS TERMINÉ                     "
  echo "==================================================================="
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
  test-all)
    test_all_endpoints
    ;;
  *)
    echo "Usage: $0 {update|direct|debug|logged|test-nodes|test-api|test-all}"
    echo ""
    echo "  update      - Met à jour le code depuis git et réinstalle"
    echo "  direct      - Lance le proxy directement avec Python pour voir les erreurs en direct"
    echo "  debug       - Lance le proxy en mode debug avec Flask"
    echo "  logged      - Lance le proxy avec logs détaillés dans logs/proxy-debug.log"
    echo "  test-nodes  - Teste la connectivité avec chaque nœud"
    echo "  test-api    - Exécute des tests basiques d'API"
    echo "  test-all    - Teste tous les endpoints API de façon exhaustive"
    ;;
esac