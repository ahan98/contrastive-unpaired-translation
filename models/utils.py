from enum import Enum

''' ENUMS '''
class NormalizationType(Enum):
    INSTANCE = 0
    BATCH = 1

class PaddingMode(Enum):
    ZEROS = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 4

''' UTIL FUNCTIONS '''

def padding_mode_to_str(paddingMode: PaddingMode) -> str:
    stringsByPaddingMode = {
        PaddingMode.ZEROS: 'zeros',
        PaddingMode.REFLECT: 'reflect',
        PaddingMode.REPLICATE: 'replicate',
        PaddingMode.CIRCULAR: 'circular'
    }

    return stringsByPaddingMode[paddingMode]