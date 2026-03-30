#!/bin/bash
# ====================================================================
# run_experiments.sh - Experimentos de referencia en CARLA Safe RL
#
# REQUISITO: El servidor CARLA debe estar corriendo antes de ejecutar.
#
#   Linux (headless):
#     ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
#
#   Windows:
#     CarlaUE4.exe -quality-level=Low
#
# Uso:
#   bash run_experiments.sh           → lanza los 3 experimentos en secuencia
#   bash run_experiments.sh baseline  → solo el experimento baseline
#   bash run_experiments.sh adaptive  → solo el experimento adaptativo
# ====================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
MAP="${MAP:-Town04}"
MAX_EPISODES="${MAX_EPISODES:-1000}"
LR="${LR:-0.0002}"

run_experiment() {
    local name="$1"
    local cmd="$2"

    echo -e "\n${BLUE}══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}▶ Experimento: ${name}${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
    echo -e "Mapa:     ${CYAN}${MAP}${NC}"
    echo -e "Comando:\n  ${YELLOW}${cmd}${NC}\n"

    eval "$cmd"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}✅ Experimento completado: ${name}${NC}"
    else
        echo -e "\n${RED}❌ Experimento falló (código ${exit_code}): ${name}${NC}"
        exit $exit_code
    fi
}

wait_for_carla() {
    echo -e "${CYAN}⏳ Esperando servidor CARLA en ${CARLA_HOST}:${CARLA_PORT}...${NC}"
    for i in $(seq 1 30); do
        if python -c "import carla; c=carla.Client('${CARLA_HOST}', ${CARLA_PORT}); c.set_timeout(2.0); c.get_server_version(); print('ok')" 2>/dev/null | grep -q ok; then
            echo -e "${GREEN}✅ CARLA disponible.${NC}\n"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    echo -e "\n${RED}❌ No se pudo conectar con CARLA. ¿Está corriendo el servidor?${NC}"
    exit 1
}

# ── Verificar CARLA ────────────────────────────────────────────────────
# wait_for_carla

FILTER="${1:-all}"

# ====================================================================
# EXPERIMENTO 1: BASELINE (Sin Shield)
# ====================================================================
if [[ "$FILTER" == "all" || "$FILTER" == "baseline" ]]; then
    run_experiment \
        "Baseline — Sin Shield" \
        "python main_train.py \
            --model_name baseline \
            --shield_type none \
            --max_episodes ${MAX_EPISODES} \
            --lr ${LR} \
            --host ${CARLA_HOST} \
            --port ${CARLA_PORT} \
            --tm_port ${TM_PORT} \
            --map ${MAP} \
            --num_npc 20 \
            --target_speed_kmh 30 \
            --success_distance 250"
fi

# ====================================================================
# EXPERIMENTO 2: SHIELD BÁSICO
# ====================================================================
if [[ "$FILTER" == "all" || "$FILTER" == "basic" ]]; then
    run_experiment \
        "Shield Básico — LIDAR + Waypoint API" \
        "python main_train.py \
            --model_name basic_shield \
            --shield_type basic \
            --max_episodes ${MAX_EPISODES} \
            --lr ${LR} \
            --host ${CARLA_HOST} \
            --port ${CARLA_PORT} \
            --tm_port ${TM_PORT} \
            --map ${MAP} \
            --num_npc 20 \
            --front_threshold 0.15 \
            --side_threshold 0.04 \
            --lateral_threshold 0.82 \
            --target_speed_kmh 30 \
            --success_distance 250"
fi

# ====================================================================
# EXPERIMENTO 3: SHIELD ADAPTATIVO (BicycleModel + Waypoint API)
# ====================================================================
if [[ "$FILTER" == "all" || "$FILTER" == "adaptive" ]]; then
    run_experiment \
        "Shield Adaptativo — BicycleModel + Waypoint API" \
        "python main_train.py \
            --model_name adaptive_shield \
            --shield_type adaptive \
            --max_episodes ${MAX_EPISODES} \
            --lr ${LR} \
            --host ${CARLA_HOST} \
            --port ${CARLA_PORT} \
            --tm_port ${TM_PORT} \
            --map ${MAP} \
            --num_npc 20 \
            --front_threshold 0.15 \
            --side_threshold 0.04 \
            --lateral_threshold 0.82 \
            --target_speed_kmh 30 \
            --success_distance 250"
fi

# ====================================================================
# RESUMEN
# ====================================================================
echo -e "\n${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ TODOS LOS EXPERIMENTOS COMPLETADOS${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"

echo -e "\nPara evaluar los modelos entrenados:\n"
echo -e "${YELLOW}python main_eval.py --model_name baseline_none_final.pth --shield_type none --map ${MAP}${NC}"
echo -e "${YELLOW}python main_eval.py --model_name basic_shield_basic_final.pth --shield_type basic --map ${MAP}${NC}"
echo -e "${YELLOW}python main_eval.py --model_name adaptive_shield_adaptive_final.pth --shield_type adaptive --map ${MAP}${NC}"

echo -e "\nPara ver las curvas de entrenamiento:"
echo -e "${CYAN}tensorboard --logdir ./runs${NC}\n"