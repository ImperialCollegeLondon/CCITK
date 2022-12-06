from enum import IntEnum


class UKBBFieldKey(IntEnum):
    # 20208: Long axis heart images - DICOM Heart MRI
    # 20209: Short axis heart images - DICOM Heart MRI
    # 20210: Aortic distensibilty images - DICOM Heart MRI
    la = 20208
    sa = 20209
    ao = 20210

print()
print(UKBBFieldKey["la"])
print()