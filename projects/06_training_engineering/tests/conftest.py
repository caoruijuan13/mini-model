from pathlib import Path
import sys


STAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(STAGE_ROOT))
for module_name in ("checkpoint", "data", "model", "optimizers", "train"):
    sys.modules.pop(module_name, None)
