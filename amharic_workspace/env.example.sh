
# Source once per shell, and at the top of each sbatch script: source env.amh.sh
export AMHARIC_DATA_DIR=$HOME/AmharicASR/amharic_workspace
export AMHARIC_CV_DIR=$AMHARIC_DATA_DIR/cv-corpus-25.0-2026-03-09/am
export AMHARIC_LEYU_JOINT_CACHE=$AMHARIC_DATA_DIR/phase2_leyu/dataset_cache_joint_independent
export AMHARIC_CACHE_DIR=$HOME/AmharicASR/clean_cache      # hub/ inside holds Whisper
export AMHARIC_CHECKPOINT_DIR=$AMHARIC_DATA_DIR/checkpoints
export AMHARIC_LOG_DIR=$AMHARIC_DATA_DIR/slurm_logs
export PYTHONPATH=$AMHARIC_DATA_DIR/scripts:${PYTHONPATH:-}
export MPLCONFIGDIR=$HOME/.cache/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1