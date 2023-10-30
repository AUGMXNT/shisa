import datasets
import glob

sampled_dataset = None
for path in map(str, glob.glob('/mnt/data/datasets/uonlp_CulturaX/ja/*.parquet')):
    dataset = datasets.Dataset.from_parquet(path).class_encode_column('source').train_test_split(
        test_size=0.01, stratify_by_column='source'
    )['test']
    if not sampled_dataset:
        sampled_dataset = dataset
    else:
        sampled_dataset = datasets.concatenate_datasets([sampled_dataset, dataset])
    print(len(sampled_dataset))
sampled_dataset.to_parquet('culturax_ja_sampled.parquet')
