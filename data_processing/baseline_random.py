import random

for split in ["train", "dev"]:
    with open(f"./baseline_full/{split}.{split}", "r") as f:
        lines = f.readlines()
        f.close()
    random.shuffle(lines)
    easy_idx = int(len(lines) * (1/3)) # 3 curricula hard coded
    medium_idx = int(len(lines) * (2/3))
    with open(f"./random_baseline/easy/{split}.{split}", "w") as f:
        easy_str = "".join([z for z in lines[:easy_idx]])
        f.write(easy_str)
        f.close()
    with open(f"./random_baseline/medium/{split}.{split}", "w") as f:
        medium_str = "".join([z for z in lines[:medium_idx]])
        f.write(medium_str)
        f.close()
    with open(f"./random_baseline/hard/{split}.{split}", "w") as f:
        hard_str = "".join([z for z in lines])
        f.write(hard_str)
        f.close()
