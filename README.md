# CS224N default final project (2022 RobustQA track)

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`

### Run test_bae
- To run test_bae.py on a particular dataset, do
```
python test_bae.py --train-dir-and-datasets="{relative_dataset_dir_path}:{dataset_name} --perturbed-data-out-path="{relative_dataset_dir_path}:{perturbed_dataset_name}""
```
e.g.
```
python test_bae.py --train-dir-and-datasets="datasets/oodomain_train:relation_extraction --perturbed-data-out-path="datasets/oodomain_train_perturbed/relation_extraction_pert32"
```

### Adam's test-bae commands: Data aug via synonyms

python test_bae.py --train-dir-and-datasets="datasets/oodomain_train:relation_extraction" --perturbed-data-out-path="datasets/oodomain_pertf_bert/relation_extraction_pertf" --num-mutations 1 --token-unmask-method="synonym" --num-indexes-upper-bound 10

python train.py --do-train --train-dir="datasets/tmp2" --train-dir-and-datasets="datasets/tmp2:race_600_3" --eval-every 2000 --run-name baseline


python test_bae.py --train-dir-and-datasets="datasets/tmp2:race_600_3" --perturbed-data-out-path="datasets/tmp2:race_600_3" --num-mutations 1 --token-unmask-method="synonym" --num-indexes-upper-bound 10 --num-data-points 600