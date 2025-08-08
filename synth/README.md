### Example Usage:

```sh
# CUDA_VISIBLE_DEVICES=2 python agmn_ygty.py --start 20 --end 31 --ds_name fikriokan/ygty --out_ds_name tst1
CUDA_VISIBLE_DEVICES=0 python agmn_ygty.py \
    --start 0 --end 5 --ds_name fikriokan/ygty \
    --out_ds_name ygty-processed-1 \
    --prompt mev_kanun_prompt
```


### L Cache:
```
CUDA_VISIBLE_DEVICES=0 python agmn_ygty.py     --start 0 --end 5 --ds_name fikriokan/ygty     --out_ds_name ygty-processed-1     --prompt mev_kanun_prompt > ygty_0_5.log 2>&1 &
```

#### Yargitay
```
CUDA_VISIBLE_DEVICES=1 python agmn_ygty.py     --start 5 --end 10 --ds_name fikriokan/ygty     --out_ds_name ygty-processed-1     --prompt mev_kanun_prompt > ygty_5_10.log 2>&1 &
```

#### Mevzuat Kanun
```
CUDA_VISIBLE_DEVICES=1 python agmn_ygty.py --start 0 --end 1 --ds_name fikriokan/sonbahcem-kan-batch-1 --out_ds_name sonbahcem-kan-batch-1-processed-1 > mev_kan.log 2>&1 &
```

#### CuKan Kanun
```
CUDA_VISIBLE_DEVICES=2 python agmn_ygty.py --start 0 --end 1 --ds_name fikriokan/sonbahcem-cukarn-batch-final --out_ds_name sonbahcem-karn-batch-1-processed-1 > mev_cukarn.log 2>&1 &
```

#### teblig Kanun
```
CUDA_VISIBLE_DEVICES=2 python agmn_ygty.py --start 0 --end 1 --ds_name fikriokan/sonbahcem-tblg-batch-1 --out_ds_name sonbahcem-tblg-batch-1-processed-1 > mev_teblg1.log 2>&1 &
```

#### krm Kanun
```
CUDA_VISIBLE_DEVICES=2 python agmn_ygty.py --start 0 --end 1 --ds_name fikriokan/sonbahcem-krm-batch-1 --out_ds_name sonbahcem-krm-batch-1-processed-1 > mev_krm1.log 2>&1 &
```

#### tuz Kanun
```
CUDA_VISIBLE_DEVICES=1 python agmn_ygty.py --start 0 --end 1 --ds_name fikriokan/sonbahcem-tuz-batch-final --out_ds_name sonbahcem-tuz-batch-1-processed-1 > mev_tuz1.log 2>&1 &
```


### NOTES
- krm'ler islenemeyecek kadar uzunlar.