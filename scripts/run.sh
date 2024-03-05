##! /bin/bash

PROMPT="Hello"
VARIANT='2b'

options="--device=cuda \
    --run_loop \
    --ckpt=/workspace/gemma/data/gemma-2b.ckpt \
    --variant="${VARIANT}" \
    --prompt="${PROMPT}""

run_cmd="python /workspace/gemma/scripts/run.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

