#!/bin/bash

echo "Запуск тестов производительности из контейнера triton-perf-client..."

run_test() {
    local model=$1
    local batch=$2
    local conc=$3
    
    echo ""
    echo "================================================="
    echo "Модель: $model, Batch: $batch, Concurrency: $conc"
    echo "================================================="
    
    docker exec triton-perf-client perf_analyzer \
        -m $model \
        -b $batch \
        --concurrency-range ${conc}:${conc}:1 \
        -i http \
        -u triton:8000 \
        --measurement-mode count_windows \
        --measurement-request-count 50
}

# Тест 1: yolo_det с разными batch sizes
run_test "yolo_det" 1 1
run_test "yolo_det" 4 1
run_test "yolo_det" 8 1

# Тест 2: yolo_seg
run_test "yolo_seg" 1 1
run_test "yolo_seg" 2 1

# Тест 3: yolo_ensemble
run_test "yolo_ensemble" 1 1

# Тест 4: Влияние concurrency на yolo_det
run_test "yolo_det" 1 2
run_test "yolo_det" 1 4
run_test "yolo_det" 1 8

echo ""
echo "Тестирование завершено!"
