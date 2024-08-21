import torch
import math
from utils.dir import create_dir

def create_example(data):
    track_id, labels, music = data
    example = {
        'features': music,
        'track_id': track_id
    }
    
    for idx, level in enumerate(labels, start=1):
        label_key = f'level{idx}'
        example[label_key] = level

    return example

def generate_pt_record(df, pt_path='val'):
    create_dir(pt_path)

    batch_size = 1024 * 50  # 50k records from each file batch
    count = 0
    total = math.ceil(len(df) / batch_size)
    for i in range(0, len(df), batch_size):
        batch_df = df[i:i + batch_size]
        pt_records = [create_example(data) for data in batch_df.values]
        path = f"{pt_path}/{str(count).zfill(10)}.pt"

        torch.save(pt_records, path)

        print(f"{count} {len(pt_records)} {path}")
        count += 1
        print(f"{count}/{total} batches / {count * batch_size} processed")

    print(f"{count}/{total} batches / {len(df)} processed")

