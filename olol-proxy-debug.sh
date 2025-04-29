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
  
  # Définir un timeout plus long pour les tests qui impliquent des LLMs
  GEN_TIMEOUT=30
  NORM_TIMEOUT=10
  
  # Utiliser le modèle disponible, avec mistral par défaut
  MODEL=$(curl -s -m $NORM_TIMEOUT http://$HOST:$PORT/api/models | jq -r '.models | keys | .[0] // "mistral:latest"')
  echo "Modèle détecté/utilisé pour les tests: $MODEL"
  
  # 1. GET /api/status - État du serveur
  echo -e "\n[TEST 1/8] GET /api/status - État du serveur"
  STATUS_RESP=$(curl -s -m $NORM_TIMEOUT http://$HOST:$PORT/api/status)
  if [ $? -eq 0 ] && [ ! -z "$STATUS_RESP" ]; then
    echo "$STATUS_RESP" | jq .
  else
    echo "❌ ERREUR: /api/status ne répond pas ou renvoie une réponse vide"
  fi
  
  # 2. GET /api/models - Liste des modèles disponibles
  echo -e "\n[TEST 2/8] GET /api/models - Liste des modèles disponibles"
  MODELS_RESP=$(curl -s -m $NORM_TIMEOUT http://$HOST:$PORT/api/models)
  if [ $? -eq 0 ] && [ ! -z "$MODELS_RESP" ]; then
    echo "$MODELS_RESP" | jq .
  else
    echo "❌ ERREUR: /api/models ne répond pas ou renvoie une réponse vide"
  fi
  
  # 3. GET /api/servers - Liste des serveurs du cluster
  echo -e "\n[TEST 3/8] GET /api/servers - Liste des serveurs du cluster"
  SERVERS_RESP=$(curl -s -m $NORM_TIMEOUT http://$HOST:$PORT/api/servers)
  if [ $? -eq 0 ] && [ ! -z "$SERVERS_RESP" ]; then
    echo "$SERVERS_RESP" | jq .
  else
    echo "❌ ERREUR: /api/servers ne répond pas ou renvoie une réponse vide"
  fi
  
  # 4. POST /api/generate - Génération de texte
  echo -e "\n[TEST 4/8] POST /api/generate - Génération de texte"
  GEN_RESP=$(curl -s -m $GEN_TIMEOUT -X POST http://$HOST:$PORT/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Explique brièvement ce qu'est un LLM en une phrase.\", \"stream\": false}")
  if [ $? -eq 0 ] && [ ! -z "$GEN_RESP" ]; then
    echo "$GEN_RESP" | jq .
    # Vérifier si la réponse contient une erreur
    if echo "$GEN_RESP" | jq -e '.error' > /dev/null; then
      echo "⚠️ AVERTISSEMENT: /api/generate a renvoyé une erreur"
    fi
  else
    echo "❌ ERREUR: /api/generate ne répond pas ou renvoie une réponse vide"
  fi
    
  # 5. POST /api/chat - Chat avec modèle
  echo -e "\n[TEST 5/8] POST /api/chat - Chat avec modèle"
  CHAT_RESP=$(curl -s -m $GEN_TIMEOUT -X POST http://$HOST:$PORT/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Bonjour, comment ça va?\"}], \"stream\": false}")
  if [ $? -eq 0 ] && [ ! -z "$CHAT_RESP" ]; then
    echo "$CHAT_RESP" | jq .
    # Vérifier si la réponse contient une erreur
    if echo "$CHAT_RESP" | jq -e '.error' > /dev/null; then
      echo "⚠️ AVERTISSEMENT: /api/chat a renvoyé une erreur"
    fi
  else
    echo "❌ ERREUR: /api/chat ne répond pas ou renvoie une réponse vide"
  fi
    
  # 6. POST /api/embeddings - Obtention d'embeddings
  echo -e "\n[TEST 6/8] POST /api/embeddings - Obtention d'embeddings"
  EMB_RESP=$(curl -s -m $NORM_TIMEOUT -X POST http://$HOST:$PORT/api/embeddings \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Ceci est un test d'embedding\"}")
  if [ $? -eq 0 ] && [ ! -z "$EMB_RESP" ]; then
    echo "$EMB_RESP" | jq .
    # Vérifier si la réponse contient une erreur
    if echo "$EMB_RESP" | jq -e '.error' > /dev/null; then
      echo "⚠️ AVERTISSEMENT: /api/embeddings a renvoyé une erreur"
    fi
  else
    echo "❌ ERREUR: /api/embeddings ne répond pas ou renvoie une réponse vide"
  fi
    
  # 7. GET /api/models/:model/context - Informations sur le contexte d'un modèle
  echo -e "\n[TEST 7/8] GET /api/models/:model/context - Informations sur le contexte"
  CTX_RESP=$(curl -s -m $NORM_TIMEOUT http://$HOST:$PORT/api/models/$MODEL/context)
  if [ $? -eq 0 ] && [ ! -z "$CTX_RESP" ]; then
    echo "$CTX_RESP" | jq .
  else
    echo "❌ ERREUR: /api/models/$MODEL/context ne répond pas ou renvoie une réponse vide"
  fi
    
  # 8. POST /api/transfer - Test de transfert de modèle
  echo -e "\n[TEST 8/8] POST /api/transfer - Test de transfert de modèle"
  # Récupérer d'abord la liste des serveurs
  if [ ! -z "$SERVERS_RESP" ]; then
    # Utiliser jq pour extraire les clés du dictionnaire 'servers' (ce sont les adresses des serveurs)
    SERVER_ADDRESSES=$(echo $SERVERS_RESP | jq -r '.servers | keys | .[0:2][]' 2>/dev/null)
    
    # Convertir en tableau
    readarray -t SERVER_ARRAY <<< "$SERVER_ADDRESSES"
    
    if [ ${#SERVER_ARRAY[@]} -ge 2 ]; then
      SOURCE_SERVER="${SERVER_ARRAY[0]}"
      TARGET_SERVER="${SERVER_ARRAY[1]}"
      
      echo "  Tentative de transfert du modèle '$MODEL' de $SOURCE_SERVER vers $TARGET_SERVER"
      TRANSFER_RESP=$(curl -s -m $GEN_TIMEOUT -X POST http://$HOST:$PORT/api/transfer \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"source\": \"$SOURCE_SERVER\", \"target\": \"$TARGET_SERVER\"}")
      
      if [ $? -eq 0 ] && [ ! -z "$TRANSFER_RESP" ]; then
        echo "$TRANSFER_RESP" | jq .
        # Vérifier si le transfert a réussi
        if echo "$TRANSFER_RESP" | jq -e '.success == true' > /dev/null; then
          echo "✅ Transfert initié avec succès"
        else
          echo "⚠️ AVERTISSEMENT: Le transfert a échoué"
        fi
      else
        echo "❌ ERREUR: /api/transfer ne répond pas ou renvoie une réponse vide"
      fi
    else
      echo "  ⚠️ Pas assez de serveurs disponibles pour tester le transfert"
    fi
  else
    echo "  ❌ Impossible de récupérer la liste des serveurs pour le test de transfert"
  fi
  
  # Résumé des tests
  echo -e "\n==================================================================="
  echo "                      RÉSUMÉ DES TESTS API                           "
  echo "==================================================================="
  echo "✅ /api/status       : $([ ! -z "$STATUS_RESP" ] && echo "OK" || echo "ÉCHEC")"
  echo "✅ /api/models       : $([ ! -z "$MODELS_RESP" ] && echo "OK" || echo "ÉCHEC")"
  echo "✅ /api/servers      : $([ ! -z "$SERVERS_RESP" ] && echo "OK" || echo "ÉCHEC")"
  echo "✅ /api/generate     : $([ ! -z "$GEN_RESP" ] && (echo "$GEN_RESP" | jq -e '.error > 0' > /dev/null 2>&1 && echo "ERREUR" || echo "OK") || echo "ÉCHEC")"
  echo "✅ /api/chat         : $([ ! -z "$CHAT_RESP" ] && (echo "$CHAT_RESP" | jq -e '.error > 0' > /dev/null 2>&1 && echo "ERREUR" || echo "OK") || echo "ÉCHEC")"
  echo "✅ /api/embeddings   : $([ ! -z "$EMB_RESP" ] && (echo "$EMB_RESP" | jq -e '.error > 0' > /dev/null 2>&1 && echo "ERREUR" || echo "OK") || echo "ÉCHEC")"
  echo "✅ /api/models/context: $([ ! -z "$CTX_RESP" ] && echo "OK" || echo "ÉCHEC")"
  echo "✅ /api/transfer     : $([ ! -z "${TRANSFER_RESP:-}" ] && (echo "${TRANSFER_RESP:-{}}" | jq -e '.success == true' > /dev/null 2>&1 && echo "OK" || echo "ERREUR") || echo "NON TESTÉ")"
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