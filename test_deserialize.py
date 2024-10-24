from ffenc_uiuc.h264_encoder import ffenc, ffdec
import numpy as np

decoder = ffdec()

# a = np.load('flashdrive/1.5/1.5-area-0.0000-3000/frame0.npy')
a = np.load('frame0.npy')
print(a)
b = decoder.process_frame(a)
print(b)