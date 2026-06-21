#python huston2013_base.py  --train_epoch 350  --gpu 0 --identity 'base' > base.out 2>&1 &
#python huston2013_mcl.py  --train_epoch 350  --gpu 0 --identity 'mcl' > mcl.out 2>&1 &
#python huston2013_dgd.py  --train_epoch 350  --gpu 0 --identity 'dgd' > dgd.out 2>&1 &
#python huston2013_gpt.py  --train_epoch 300  --gpu 0 --identity 'gpt' > gpt.out 2>&1 &
#python huston2013_dw.py  --train_epoch 300  --gpu 0 --identity 'dw' > dw.out 2>&1 &
#python huston2013_mcl_se.py  --train_epoch 300  --gpu 0 --identity 'mcl_se' > mcl_se.out 2>&1 &
#python huston2013_mcl_drop.py  --train_epoch 300  --gpu 0 --identity 'mcl_drop' > mcl_drop.out 2>&1 &


python huston2013_dw_prune.py  --train_epoch 400  --gpu 0 --osc 0.8 --identity 'dw_prune8' > dwp8.out 2>&1 &
python huston2013_dw_prune.py  --train_epoch 400  --gpu 0 --osc 0.6 --identity 'dw_prun6' > dwp6.out 2>&1 &
python huston2013_dw_prune.py  --train_epoch 400  --gpu 1 --osc 0.4 --identity 'dw_prun4' > dwp4.out 2>&1 &
python huston2013_dw_prune.py  --train_epoch 400  --gpu 1 --osc 0.2 --identity 'dw_prun2' > dwp2.out 2>&1 &
python huston2013_dw_prune.py  --train_epoch 400  --gpu 1 --osc 0.9 --identity 'dw_prun9' > dwp9.out 2>&1 &
python huston2013_dw_prune.py  --train_epoch 400  --gpu 0 --l1_loss 0.001 --osc 0.8 --identity 'dw_prune81' > dwp81.out 2>&1 &
python huston2013_dw_prune.py  --train_epoch 400  --gpu 0 --l1_loss 0.01 --osc 0.8 --identity 'dw_prune81' > dwp81.out 2>&1 &
python huston2013_dw_prune1.py  --train_epoch 400  --gpu 0 --l1_loss 0.01 --osc 0.9 --identity 'dw_prune1' > dwp1.out 2>&1 &
python huston2013_dw_prune2.py  --train_epoch 400  --gpu 0 --l1_loss 0.01 --osc 0.9 --identity 'dw_prune2' > dwp2.out 2>&1 &

python huston2013_gate.py  --train_epoch 400  --gpu 0 --identity 'gate1' > gate1.out 2>&1 &
python huston2013_gate.py  --train_epoch 400  --gpu 0 --identity 'gate2' > gate2.out 2>&1 &
python huston2013_gate.py --lr_decrease 'multi_step' --train_epoch 400  --gpu 0 --identity 'gate3' > gate3.out 2>&1 &
python huston2013_gate1.py --lr_decrease 'multi_step' --train_epoch 400  --gpu 1 --identity 'gate4' > gate4.out 2>&1 &