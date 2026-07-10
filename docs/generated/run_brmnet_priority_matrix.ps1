$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
Set-Location ..

# priority 1: full_budget0p8_seed0 - Full BRM-Net: budget gates + compact export + availability-aware modality dropout.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 0 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\full'

# priority 2: without_modality_dropout_budget0p8_seed0 - Tests whether missing-modality robustness comes from dropout training.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 0 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.0 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_modality_dropout'

# priority 3: budget_only_budget0p8_seed0 - Compression mechanism without explicit missing-modality training.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 0 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.0 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\budget_only'

# priority 4: without_budget_loss_budget0p8_seed0 - Reliability/dropout behavior without an active budget penalty.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 0 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 0.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_budget_loss'

# priority 5: without_reliability_uniform_fusion_budget0p8_seed0 - Ablates reliability weighting by averaging available modalities.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 0 --target-budget 0.8 --gate-type hard_concrete --fusion-mode uniform --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_reliability_uniform_fusion'

# priority 6: legacy_sigmoid_reference_budget0p8_seed0 - Legacy gate reference; not a deployable hard-concrete export baseline.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 0 --target-budget 0.8 --gate-type legacy_sigmoid --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\legacy_sigmoid_reference'

# priority 7: full_budget0p8_seed1 - Full BRM-Net: budget gates + compact export + availability-aware modality dropout.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 1 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\full'

# priority 8: without_modality_dropout_budget0p8_seed1 - Tests whether missing-modality robustness comes from dropout training.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 1 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.0 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_modality_dropout'

# priority 9: budget_only_budget0p8_seed1 - Compression mechanism without explicit missing-modality training.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 1 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.0 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\budget_only'

# priority 10: without_budget_loss_budget0p8_seed1 - Reliability/dropout behavior without an active budget penalty.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 1 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 0.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_budget_loss'

# priority 11: without_reliability_uniform_fusion_budget0p8_seed1 - Ablates reliability weighting by averaging available modalities.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 1 --target-budget 0.8 --gate-type hard_concrete --fusion-mode uniform --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_reliability_uniform_fusion'

# priority 12: legacy_sigmoid_reference_budget0p8_seed1 - Legacy gate reference; not a deployable hard-concrete export baseline.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 1 --target-budget 0.8 --gate-type legacy_sigmoid --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\legacy_sigmoid_reference'

# priority 13: full_budget0p8_seed2 - Full BRM-Net: budget gates + compact export + availability-aware modality dropout.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 2 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\full'

# priority 14: without_modality_dropout_budget0p8_seed2 - Tests whether missing-modality robustness comes from dropout training.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 2 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.0 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_modality_dropout'

# priority 15: budget_only_budget0p8_seed2 - Compression mechanism without explicit missing-modality training.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 2 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.0 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\budget_only'

# priority 16: without_budget_loss_budget0p8_seed2 - Reliability/dropout behavior without an active budget penalty.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 2 --target-budget 0.8 --gate-type hard_concrete --fusion-mode reliability --lambda-budget 0.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_budget_loss'

# priority 17: without_reliability_uniform_fusion_budget0p8_seed2 - Ablates reliability weighting by averaging available modalities.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 2 --target-budget 0.8 --gate-type hard_concrete --fusion-mode uniform --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\without_reliability_uniform_fusion'

# priority 18: legacy_sigmoid_reference_budget0p8_seed2 - Legacy gate reference; not a deployable hard-concrete export baseline.
'D:\software\anaconda3\envs\hslinets\python.exe' scripts/run_brmnet_houston.py --data-root 'D:\Academic\HSLiNets-main\Dataset' --data-format raw --pair-modalities hsi+lidar --split-protocol official --split-seed 42 --seed 2 --target-budget 0.8 --gate-type legacy_sigmoid --fusion-mode reliability --lambda-budget 1.0 --modality-dropout-prob 0.25 --epochs 20 --compact-finetune-epochs 10 --batch-size 32 --budget-metric macs --output-dir 'output\experiments_priority\legacy_sigmoid_reference'
