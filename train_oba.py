import subprocess
import sys

# sys.executable = tacno onaj Python koji trenutno radi (iz venv-a)
print(">>> Trening 1/2: regresija")
subprocess.run([sys.executable, "train.py"])

print(">>> Trening 2/2: klasifikacija (balansirana)")
subprocess.run([sys.executable, "train_klasifikacija.py"])

print(">>> Gotovo — oba modela istrenirana.")