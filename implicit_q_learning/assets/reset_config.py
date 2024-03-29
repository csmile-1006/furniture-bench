import os
from typing import Any, Dict

import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat


reset_config: Dict[str, Any] = {
    "one_leg": {
        {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }
    }
}
