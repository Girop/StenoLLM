from datasets import Dataset

failed = 0

ds = Dataset.from_json("test.json")
for item in ds:
    failed += (mathced := item['secret'] != item['decoded'])
    print(item["demonstration"], not mathced)
    print("=" * 20)

print("Failed: ", failed)

