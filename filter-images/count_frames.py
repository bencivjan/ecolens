from pathlib import Path

root_dir = Path("JH/1.5")

print('Frequency,Filter,Threshold,Frame Bitrate,Frame Count')

for directory in sorted([d for d in root_dir.iterdir() if d.is_dir()]):
    freq, filter, threshold, bitrate = directory.name.split('-')
    num_files = len([f for f in directory.iterdir() if f.is_file()])
    print(f"{freq},{filter},{threshold},{bitrate},{num_files}")
