import pybullet_data
from pathlib import Path

data_path = Path(pybullet_data.getDataPath())
print(f"PyBullet Data Path: {data_path}")

racecar_path = data_path / "racecar.urdf" # racecar.urdf 通常直接在 pybullet_data 目录下
if racecar_path.exists():
    print(f"Found 'racecar.urdf' at: {racecar_path}")
else:
    print(f"ERROR: 'racecar.urdf' NOT found at {racecar_path}")
    print("Files and directories in pybullet_data path:")
    for item in data_path.iterdir():
        print(f"  - {item.name}")