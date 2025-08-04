#!/bin/bash

# Sampling
TASK=retrosynthesis
GRAPH_TOWER=moleculestm
EPOCH=20
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path "../checkpoints/Graph-LLaVA/retrosynthesis/checkpoint/" \
    --in-file "../MolInstruction/Molecule-oriented_Instructions/retrosynthesis_test.json" \
    --answers-file eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep-vlora4.jsonl \
    --graph-checkpoint-path ../MoleculeSTM/molecule_model.pth \
    --model-base ../vicuna-7b \
    --batch_size 1 --temperature 0.1 --top_p 1.0 \
    --add-selfies \
    --debug
## Calculate the 'BLEU', 'exact match score', 'Levenshtein score' and 'validity'
python -m llava.eval.molecule_metrics.mol_translation_selfies \
    --input_file=eval_result/${GRAPH_TOWER}-${TASK}-${EPOCH}ep-vlora4.jsonl
## Calculate the 'MACCS', 'RDK' and 'Morgan' similarity
python -m llava.eval.molecule_metrics.fingerprint_metrics \
    --input_file=eval_result/${GRAPH_TOWER}-${TASK}-${EPOCH}ep-vlora4.jsonl