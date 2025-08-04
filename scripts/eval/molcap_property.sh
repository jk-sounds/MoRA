#!/bin/bash
# Sampling
TASK=property_pred
GRAPH_TOWER=moleculestm
EPOCH=5
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path "../checkpoints/Graph-LLaVA/property/checkpoint/" \
    --in-file "../data/property_pred/property_pred_test.json" \
    --answers-file eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl \
    --graph-checkpoint-path ../MoleculeSTM/molecule_model.pth \
    --model-base ../vicuna-7b \
    --batch_size 1 --temperature 0.2 --top_p 1.0 \
    --add-selfies \
    --debug 
# Evaluation
python -m llava.eval.molecule_metrics.property_metrics \
    --eval_result_file eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl