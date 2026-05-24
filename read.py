from datasets import Dataset

ds = Dataset.load_from_disk("steno_dataset")
for item in ds:
    print(item['secret'], item['decoded'], item['secret'] == item['decoded'])

