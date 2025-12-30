import subprocess
import sys

def install_requirements(file_path='requirements.txt'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", file_path])

if __name__ == "__main__":
    install_requirements()
