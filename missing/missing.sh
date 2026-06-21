python huston2013_gate.py  --train_epoch 400  --gpu 0 --identity 'gate1' > gate1.out 2>&1 &
python huston2013_gate1.py  --train_epoch 400  --gpu 0 --identity 'gate2' > gate2.out 2>&1 &
python huston2013_gate.py --lr_decrease 'multi_step' --train_epoch 400  --gpu 0 --identity 'gate3' > gate3.out 2>&1 &
python huston2013_gate1.py --lr_decrease 'multi_step' --train_epoch 400  --gpu 1 --identity 'gate4' > gate4.out 2>&1 &

python huston2013_gate2.py  --train_epoch 400  --gpu 0 --identity 'gate3' > gate3.out 2>&1 &
python huston2013_gate2.py  --lr_decrease 'multi_step' --train_epoch 400  --gpu 0 --identity 'gate3_step' > gate3_step.out 2>&1 &
python huston2013_gate2.py  --train_epoch 400  --gpu 0 --identity 'gate4'
python huston2013_gate2.py  --train_epoch 400  --gpu 0 --identity 'gate4' > gate4.out 2>&1 &
python huston2013_gate2.py  --lr_decrease 'multi_step' --train_epoch 400  --gpu 0 --identity 'gate4_step' > gate4_step.out 2>&1 &



python huston2013_gate.py  --train_epoch 30  --gpu 0 --identity 'missing'  > missing.out 2>&1 &
python main_base.py  --train_epoch 300  --gpu 0 --identity 'base' > base.out 2>&1 &
python main_base.py  --train_epoch 300  --gpu 0 --identity 'base' > base.out 2>&1 &
python main_base_sfd.py  --train_epoch 30  --gpu 0  --loss_num 4 --identity  'base_sfd'
python main_base_sfd.py  --train_epoch 300  --gpu 0  --identity  'base_sfd' > base_sfd.out 2>&1 &


python main.py  --train_epoch 300   --gpu 0  --identity  'sfd_no_loss' > sfd_no_loss.out 2>&1 &


python main.py  --train_epoch 300   --gpu 1  --identity  'sfd_md_loss1' > sfd_md_loss1.out 2>&1 &

python main.py  --train_epoch 300   --gpu 1  --identity  'true_base'  > true_base.out 2>&1 &


python main.py  --train_epoch 300   --gpu 0  --identity  'sfd_no_loss' > sfd_no_loss.out 2>&1 &

python main.py  --train_epoch 300   --gpu 1  --identity  'sfd_md_loss2' > sfd_md_loss2.out 2>&1 &
python main.py  --train_epoch 300   --gpu 1  --vis_interval 50 --identity  'figure_stable' > figure_stable.out 2>&1 &
python main.py  --train_epoch 300   --gpu 1  --vis_interval 50 --identity  'figure_para1' > figure_para1.out 2>&1 &


python main.py  --train_epoch 300   --gpu 0  --identity  'ckd1' > ckd1.out 2>&1 &
python main.py  --train_epoch 300   --gpu 0  --identity 'ckd3' --threshold 0.3 > ckd3.out 2>&1 &


python main.py  --train_epoch 300   --gpu 0  --identity  'nopool' > no_pool.out 2>&1 &

python main.py  --train_epoch 300   --gpu 0  --identity  'depose' > depose.out 2>&1 &

python main.py  --train_epoch 300 --class_num 8 --data_root 'Berlin'  --pair_modalities 'hsi1+sar'  --gpu 0  --identity  'depose_berlin' > depose_berlin.out 2>&1 &


python main.py  --train_epoch 300   --gpu 0  --identity  'depose_lidar_lambda' --lidar_lambda 4.0 > depose_lidar_lambda.out 2>&1 &


python main.py  --train_epoch 300   --gpu 0  --identity  'drfuse' > drfuse.out 2>&1 &