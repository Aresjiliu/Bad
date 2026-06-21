python huston2013_gate.py  --train_epoch 400  --gpu 0 --identity 'gate1' > gate1.out 2>&1 &
python huston2013_gate1.py  --train_epoch 400  --gpu 0 --identity 'gate2' > gate2.out 2>&1 &
python huston2013_gate.py --lr_decrease 'multi_step' --train_epoch 400  --gpu 0 --identity 'gate3' > gate3.out 2>&1 &
python huston2013_gate1.py --lr_decrease 'multi_step' --train_epoch 400  --gpu 1 --identity 'gate4' > gate4.out 2>&1 &

python huston2013_gate2.py  --train_epoch 400  --gpu 0 --identity 'gate3' > gate3.out 2>&1 &
python huston2013_gate2.py  --lr_decrease 'multi_step' --train_epoch 400  --gpu 0 --identity 'gate3_step' > gate3_step.out 2>&1 &
python huston2013_gate2.py  --train_epoch 400  --gpu 0 --identity 'gate4'
python huston2013_gate2.py  --train_epoch 400  --gpu 0 --identity 'gate4' > gate4.out 2>&1 &
python huston2013_gate2.py  --lr_decrease 'multi_step' --train_epoch 400  --gpu 0 --identity 'gate4_step' > gate4_step.out 2>&1 &



python huston2013_gate.py  --train_epoch 30  --gpu 0 --identity 'missing'