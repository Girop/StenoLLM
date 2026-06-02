from datasets import Dataset

failed = 0

ds = Dataset.from_json("test.json")
for item in ds:
    failed += (matched := item['secret'] != item['decoded'])
    print(item["demonstration"], matched)
    print("=" * 20)

print("Failed: ", failed)

