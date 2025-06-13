#!/bin/bash

stage=$1
stop_stage=$2
model=$3 # F5TTS_v1_Base

if [ -z "$model" ]; then
    echo "Model is none, using default model F5TTS_v1_Base"
    model=F5TTS_v1_Base
fi
echo "Start stage: $stage, Stop stage: $stop_stage, Model: $model"
export CUDA_VISIBLE_DEVICES=0

F5_TTS_HF_DOWNLOAD_PATH=./F5-TTS
F5_TTS_TRT_LLM_CHECKPOINT_PATH=./trtllm_ckpt
F5_TTS_TRT_LLM_ENGINE_PATH=./f5_trt_llm_engine

vocoder_trt_engine_path=vocos_vocoder.plan
model_repo=./model_repo

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading f5 tts from huggingface"
    huggingface-cli download SWivid/F5-TTS --local-dir $F5_TTS_HF_DOWNLOAD_PATH
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Converting SafeTensors to PyTorch format"
    python3 -c "
import torch
from safetensors.torch import load_file
import os

model_path = '$F5_TTS_HF_DOWNLOAD_PATH/$model'
safetensors_file = os.path.join(model_path, 'model_1250000.safetensors')
pt_file = os.path.join(model_path, 'model_1250000.pt')

if os.path.exists(safetensors_file):
    print(f'Converting {safetensors_file} to {pt_file}')
    state_dict = load_file(safetensors_file)
    torch.save(state_dict, pt_file)
    print('Conversion completed successfully')
else:
    print(f'SafeTensors file not found at {safetensors_file}')
"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Converting checkpoint"
    python3 ./scripts/convert_checkpoint.py \
        --timm_ckpt "$F5_TTS_HF_DOWNLOAD_PATH/$model/model_1250000.pt" \
        --output_dir "$F5_TTS_TRT_LLM_CHECKPOINT_PATH" --model_name $model
    python_package_path=/usr/local/lib/python3.12/dist-packages
    cp -r patch/* $python_package_path/tensorrt_llm/models
    trtllm-build --checkpoint_dir $F5_TTS_TRT_LLM_CHECKPOINT_PATH \
      --max_batch_size 8 \
      --output_dir $F5_TTS_TRT_LLM_ENGINE_PATH --remove_input_padding disable
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Exporting vocos vocoder"
    onnx_vocoder_path=vocos_vocoder.onnx
    python3 scripts/export_vocoder_to_onnx.py --vocoder vocos --output-path $onnx_vocoder_path
    bash scripts/export_vocos_trt.sh $onnx_vocoder_path $vocoder_trt_engine_path
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Building triton server"
    rm -rf $model_repo
    cp -r ./model_repo_f5_tts $model_repo
    # FIXED: Proper single-string argument format for fill_template.py
    python3 scripts/fill_template.py -i $model_repo/f5_tts/config.pbtxt \
        "vocab:$F5_TTS_HF_DOWNLOAD_PATH/$model/vocab.txt,model:$F5_TTS_HF_DOWNLOAD_PATH/$model/model_1250000.pt,trtllm:$F5_TTS_TRT_LLM_ENGINE_PATH,vocoder:vocos"
    cp $vocoder_trt_engine_path $model_repo/vocoder/1/vocoder.plan
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Starting triton server"
    tritonserver --model-repository=$model_repo
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Testing triton server"
    num_task=1
    log_dir=./log_concurrent_tasks_${num_task}
    rm -rf $log_dir
    python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts --log-dir $log_dir
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "Testing http client"
    
    # Check for reference audio in multiple locations
    audio_paths=(
        "../../infer/examples/basic/basic_ref_en.wav"
        "./examples/basic_ref_en.wav"
        "$F5_TTS_HF_DOWNLOAD_PATH/examples/basic_ref_en.wav"
        "./basic_ref_en.wav"
        "./test_reference.wav"
    )
    
    audio=""
    for path in "${audio_paths[@]}"; do
        if [ -f "$path" ]; then
            audio="$path"
            echo "Found reference audio at: $audio"
            break
        fi
    done
    
    if [ -z "$audio" ]; then
        echo "Warning: No reference audio found. Creating a dummy one..."
        # Create a simple test audio if none exists
        python3 -c "
import torch
import torchaudio
import numpy as np

# Generate a simple sine wave for testing
sample_rate = 24000
duration = 3  # seconds
frequency = 440  # Hz

t = np.linspace(0, duration, int(sample_rate * duration), False)
waveform = np.sin(2 * np.pi * frequency * t)
waveform = torch.from_numpy(waveform).float().unsqueeze(0)

torchaudio.save('test_reference.wav', waveform, sample_rate)
print('Created test reference audio: test_reference.wav')
"
        audio="test_reference.wav"
    fi
    
    reference_text="Some call me nature, others call me mother nature."
    target_text="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring."
    
    echo "Using reference audio: $audio"
    echo "Reference text: $reference_text"
    echo "Target text: $target_text"
    
    python3 client_http.py --reference-audio "$audio" --reference-text "$reference_text" --target-text "$target_text"
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "TRT-LLM: offline decoding benchmark test"
    batch_size=1
    split_name=wenetspeech4tts
    backend_type=trt
    log_dir=./log_benchmark_batch_size_${batch_size}_${split_name}_${backend_type}
    rm -rf $log_dir
    ln -sf model_repo_f5_tts/f5_tts/1/f5_tts_trtllm.py ./
    torchrun --nproc_per_node=1 \
    benchmark.py --output-dir $log_dir \
    --batch-size $batch_size \
    --enable-warmup \
    --split-name $split_name \
    --model-path $F5_TTS_HF_DOWNLOAD_PATH/$model/model_1250000.pt \
    --vocab-file $F5_TTS_HF_DOWNLOAD_PATH/$model/vocab.txt \
    --vocoder-trt-engine-path $vocoder_trt_engine_path \
    --backend-type $backend_type \
    --tllm-model-dir $F5_TTS_TRT_LLM_ENGINE_PATH || exit 1
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    echo "Native Pytorch: offline decoding benchmark test"
    pip install -r requirements-pytorch.txt
    batch_size=1
    split_name=wenetspeech4tts
    backend_type=pytorch
    log_dir=./log_benchmark_batch_size_${batch_size}_${split_name}_${backend_type}
    rm -rf $log_dir
    ln -sf model_repo_f5_tts/f5_tts/1/f5_tts_trtllm.py ./
    torchrun --nproc_per_node=1 \
    benchmark.py --output-dir $log_dir \
    --batch-size $batch_size \
    --split-name $split_name \
    --enable-warmup \
    --model-path $F5_TTS_HF_DOWNLOAD_PATH/$model/model_1250000.pt \
    --vocab-file $F5_TTS_HF_DOWNLOAD_PATH/$model/vocab.txt \
    --backend-type $backend_type \
    --tllm-model-dir $F5_TTS_TRT_LLM_ENGINE_PATH || exit 1
fi
