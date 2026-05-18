import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from config import get_config_dict


def test_get_config_dict():
    cfg = get_config_dict()
    assert isinstance(cfg, dict)
    assert 'HOST' in cfg
    assert 'PORT' in cfg
