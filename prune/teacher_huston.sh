python huston2013_multi_share_unimodal_center.py  --train_epoch 300 --data_root "Berlin"  --class_num 8 --pair_modalities "hsi1+sar" --gpu 0 --l1_loss 0.001 --gama 1.01 --osc 0.9 > berlin9.out 2>&1 &
python huston2013_multi_share_unimodal_center.py  --train_epoch 300 --data_root "Berlin"  --class_num 8 --pair_modalities "hsi1+sar" --gpu 1 --l1_loss 0.001 --gama 1.01 --osc 0.7  > berlin7.out 2>&1 &
python huston2013_multi_share_unimodal_center.py  --train_epoch 300 --data_root "Berlin"  --class_num 8 --pair_modalities "hsi1+sar" --gpu 0 --l1_loss 0.001 --gama 1.01 --osc 0.5  > berlin5.out 2>&1 &
python huston2013_multi_share_unimodal_center.py  --train_epoch 300 --data_root "Berlin"  --class_num 8 --pair_modalities "hsi1+sar" --gpu 1 --l1_loss 0.001 --gama 1.01 --osc 0.1  > berlin1.out 2>&1 &
python huston2013_multi_share_unimodal_center.py  --train_epoch 300 --data_root "Berlin"  --class_num 8 --pair_modalities "hsi1+sar" --gpu 0 --l1_loss 0.001 --gama 1.01 --osc 0.05  > berlin05.out 2>&1 &





