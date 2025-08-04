#!/bin/bash

# Sampling
TASK=molcap
GRAPH_TOWER=moleculestm
EPOCH=10
OUT_FILE=eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep-final-molcap.jsonl
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path "../checkpoints/Graph-LLaVA/molcap/checkpoint/" \
    --in-file "../data/desc_pred/molecular_desc_test_new.json" \
    --answers-file $OUT_FILE \
    --graph-checkpoint-path ../model/MoleculeSTM/molecule_model.pth \
    --model-base ../model/vicuna-2-7b-extent \
    --batch_size 1 --temperature 0.1 --top_p 1.0 \
    --add-selfies \
    --debug
## Calculate the 'MACCS', 'RDK' and 'Morgan' similarity
python -m llava.eval.eval_molcap --molcap_result_file $OUT_FILE \
    --text2mol_bert_path ../model/scibert