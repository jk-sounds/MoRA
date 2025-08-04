#!/bin/bash

# Sampling
TASK=reagent_pred
GRAPH_TOWER=moleculestm
EPOCH=10
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path "../checkpoints/Graph-LLaVA/reagent/checkpoint/" \
    --in-file "../data/reagent_pred/reagent_pred_test.json" \
    --answers-file eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep-final-FPN-reagent.jsonl \
    --graph-checkpoint-path ../model/MoleculeSTM/molecule_model.pth \
    --model-base ../model/vicuna-2-7b-extent \
    --batch_size 1 --temperature 0.1 --top_p 1.0 \
    --add-selfies \
    --debug
## Calculate the 'BLEU', 'exact match score', 'Levenshtein score' and 'validity'
python -m llava.eval.molecule_metrics.mol_translation_selfies \
    --input_file=eval_result/${GRAPH_TOWER}-${TASK}-${EPOCH}ep-final-FPN-reagent.jsonl
## Calculate the 'MACCS', 'RDK' and 'Morgan' similarity
python -m llava.eval.molecule_metrics.fingerprint_metrics \
    --input_file=eval_result/${GRAPH_TOWER}-${TASK}-${EPOCH}ep-final-FPN-reagent.jsonl