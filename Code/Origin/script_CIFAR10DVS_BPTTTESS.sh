

# TESS DVS Gesture
CUDA_VISIBLE_DEVICES=0 python main.py --dataset DVSGesture --arch dvs_tessvgg_model --data-path ~/Datasets --save-path ./experiments/DVSGesture_TESS_ --trials 1 --epochs 200 --batch-size 16 --val-batch-size 16 --print-freq 20 --delay-ls 20 --factors-stdp 0.2 0.5 0 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSGesture" --training-mode tess --loss "CE" --wn --optimizer Adam

# BPTT DVS Gesture
CUDA_VISIBLE_DEVICES=2 python main.py --dataset DVSGesture --arch dvs_vgg_bptt --data-path ~/Datasets --save-path ./experiments/DVSGesture_BPTT_ --trials 1 --epochs 200 --batch-size 16 --val-batch-size 16 --print-freq 20 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSGesture" --training-mode bptt --loss "CE" --wn --optimizer Adam