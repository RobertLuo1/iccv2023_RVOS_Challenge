#!/usr/bin/env bash
set -x

GPUS=${GPUS:-1}
PORT=${PORT:-29500}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

OUTPUT_DIR=$1
PRETRAINED_WEIGHTS=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

# train
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# inference
# CHECKPOINT=${OUTPUT_DIR}/ytvos_video_swin_base_joint.pth
python3 inference_ytvos_for_test.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --resume=${PRETRAINED_WEIGHTS} ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"