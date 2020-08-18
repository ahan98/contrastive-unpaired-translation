from enum import Enum

''' ENUMS '''
class NormType(Enum):
    NONE = 0
    INSTANCE = 1
    BATCH = 2

class PaddingMode(Enum):
    ZEROS = 0
    REFLECT = 1
    REPLICATE = 2
    CIRCULAR = 3

class ActivationType(Enum):
    NONE = 0
    RELU = 1
    TANH = 2
    LEAKY = 3

''' UTIL FUNCTIONS '''

def padding_mode_to_str(paddingMode: PaddingMode) -> str:
    stringsByPaddingMode = {
        PaddingMode.ZEROS: 'zeros',
        PaddingMode.REFLECT: 'reflect',
        PaddingMode.REPLICATE: 'replicate',
        PaddingMode.CIRCULAR: 'circular'
    }

    return stringsByPaddingMode[paddingMode]
