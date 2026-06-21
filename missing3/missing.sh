

python main.py  --train_epoch 300   --gpu 1  --identity  'res0'  --use_id_loss False --use_intra_loss False  --use_drfuse_loss False > res0.out 2>&1 &
python main.py  --train_epoch 300   --gpu 1  --identity  'res1'  --use_intra_loss False  --use_drfuse_loss False > res1.out 2>&1 &

python main.py  --train_epoch 300   --gpu 0  --identity  'res2'  --use_md_loss False --use_intra_loss False > res2.out 2>&1 &

python main.py  --train_epoch 300   --gpu 0  --identity  'res3'  --use_md_loss False  > res3.out 2>&1 &




python main.py  --train_epoch 300   --gpu 1  --identity  'res0'  --use_md_loss True > res0.out 2>&1 &

python main.py  --train_epoch 300   --gpu 0  --identity  'res1'  --use_md_loss True --use_id_loss True > res1.out 2>&1 &

python main.py  --train_epoch 300   --gpu 0  --identity  'res2'  --use_id_loss True --use_drfuse_loss True > res2.out 2>&1 &

python main.py  --train_epoch 300   --gpu 0  --identity  'res3'  --use_id_loss True --use_intra_loss True  --use_drfuse_loss True > res3.out 2>&1 &


python main.py --train_epoch 300 --gpu 0 --identity 'improved_mdfuse' \
      --use_md_loss True --use_drfuse_loss False --use_id_loss False --use_intra_loss False \
      --lambda0 0.8 --lambda1 0.3 --lambda2 0.2 --lidar_lambda 0.8 > res0.out 2>&1 &