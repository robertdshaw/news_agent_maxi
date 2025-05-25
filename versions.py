from pathlib import Path

out = Path("model_output")
v1 = out / "v1"
v1.mkdir(exist_ok=True)

for file in out.iterdir():
    if file.is_file() and file.suffix in {".pkl", ".csv", ".png"}:
        new_name = f"{file.stem}_1{file.suffix}"
        file.rename(v1 / new_name)
print(f"Moved and renamed {len(list(v1.iterdir()))} files into {v1}")
