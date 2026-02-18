from datasets import load_dataset

dataset = load_dataset("davanstrien/WELFake")

print(dataset["train"][0])
dataset = load_dataset(
    "davanstrien/WELFake",
    cache_dir="./data"
)