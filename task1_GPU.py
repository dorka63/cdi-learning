import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import njit,prange
import cProfile
import pstats
import cupy as cp

try:
    import secrets
    use_secrets = True
except ImportError:
    import np.random
    use_secrets = False

IMG_PATH='eo2_rot.jpg'
crypt = Image.open(IMG_PATH).convert('L')

original_width, original_height = crypt.size

scale_factor = 4
new_size = (original_width // scale_factor, original_height // scale_factor)
resized_image = crypt.resize(new_size, Image.Resampling.LANCZOS)

canvas_size = (555, 555)
canvas = Image.new('L', canvas_size, color=0)

x_offset = (canvas_size[0] - new_size[0]) // 2
y_offset = (canvas_size[1] - new_size[1]) // 2

canvas.paste(resized_image, (x_offset, y_offset))
canvas.save('output_image.jpg')

def show_2_images(data1, data2, title1="Module", title2="Phase", cmap1=None, cmap2=None):
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray) and data1.shape == data2.shape:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title(title1)
        if data1.ndim == 3 and data1.shape[2] == 3:
            plt.imshow(data1)
        else:
            plt.imshow(data1, cmap=cmap1)
            plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title(title2)
        if data2.ndim == 3 and data2.shape[2] == 3:
            plt.imshow(data2)
        else:
            plt.imshow(data2, cmap=cmap2)
            plt.colorbar()

        plt.tight_layout()
        plt.show()
    else:
        print("Изображения должны быть numpy массивами одинакового размера.")


def img_prep(i):
    pixel_values = np.array(i, dtype=np.complex64)
    pixel_values = pixel_values[:, :, 0] + pixel_values[:, :, 1] * 256 + pixel_values[:, :, 2] * 256 * 256
    return pixel_values

def img_unprep(pixel_values):
    max_val = np.max(pixel_values)
    scaled_values = (255 * (pixel_values) / (max_val)).astype(np.uint8)
    height, width = scaled_values.shape
    bw_image = Image.fromarray(scaled_values, mode='L')
    return bw_image

center_square_start = (555 // 2) - (555 // 4)
center_square_end = (555 // 2) + (555 // 4)

STRICT_MASK = np.zeros((555, 555))
STRICT_MASK[center_square_start:center_square_end, center_square_start:center_square_end] = 1.0

mask = STRICT_MASK
antimask = 1.0 - STRICT_MASK

@njit(parallel=True)
def generate_random_complex_field(width=555, height=555):
    rand_field = np.empty((height, width), dtype=np.complex64)
    for i in prange(height):
        for j in range(width):
            amplitude = np.random.randint(0, 2**24-1)
            phase = np.random.random() * 2 * np.pi
            rand_field[i, j] = amplitude * np.exp(1j * phase)
    return rand_field

def generate_secret_random_complex_field(width=555, height=555):
    rand_field = np.empty((height, width), dtype=np.complex64)
    amplitudes = np.empty((height, width))
    phases = np.empty((height, width))
    for i in prange(height):
        for j in range(width):
            amplitudes[i,j] = secrets.randbelow(2**24-1)
            phases[i,j] = secrets.randbits(32) / 2**32 * 2 * np.pi
    rand_field = amplitudes * np.exp(1j * phases)
    return rand_field

def FT(data):
    data_gpu = cp.asarray(data)
    result_gpu = cp.fft.fft2(data_gpu)
    return result_gpu
def IFT(data):
  data_gpu = cp.asarray(data)
  result_gpu = cp.fft.ifft2(data_gpu)
  return cp.asnumpy(result_gpu)

def steps(X_inp, X_source):
    return IFT(cp.asarray(X_source) * cp.exp(1j * cp.angle(FT(X_inp))))

def apply_mask2field(mask,field):
  mask=cp.asarray(mask)
  field=cp.asarray(field)
  mod=(cp.abs(field)*mask)
  phase=(cp.angle(field)*mask)
  return(cp.asnumpy(mod*np.exp(1j*phase)))

def ER(N_iter: int, target, source):
    A = target  # 0th iteration --- random field distribution or the previous result
    for i in range(N_iter):
        D = steps(A, source)
        #A = D * mask  # r-domain
        A=apply_mask2field(mask,D)
    D_norm = cp.linalg.norm(D)
    A_norm = cp.linalg.norm(A)
    Error = cp.sqrt(D_norm**2 - A_norm**2) / D_norm
    return [A, Error]

def HIO(N_iter: int, beta: float, Target, Source):
    A = Target  # 0th iteration --- random field distribution or the previous result
    for i in range(N_iter):
        D = steps(A, Source)
        A = apply_mask2field(mask,D) + apply_mask2field(antimask, (A - beta * D))  # r-domain
    return A


#from the task (30 HIO +10 ER)
def retrieving(img, real_img, beta):
    #real_img is the r-domain field for the HIO's 0th iteration, img --- "true" reciporal magnitude
    c_hio = HIO(N_iter=30, beta=beta, Target=real_img, Source=img)
    #got a c_hio field (in r-domain), now put it into 0th iteration for ER
    c_er,err = ER(10, target=c_hio, source=img)
    return [c_er,err]

# This algorithm is taken from the work:
# Artyukov, I.A., Vinogradov, A.V., Gorbunkov, M.V. et al.
# Virtual Lens Method for Near-Field Phase Retrieval.
# Bull. Lebedev Phys. Inst. 50, 414–419 (2023).
# https://doi.org/10.3103/S1068335623100020

def retr_block(inp):
    out1,err1 = retrieving(img=crypt_values,real_img=inp, beta=1.)
    out2,err2 = retrieving(img=crypt_values,real_img=out1,beta=0.7)
    out3,err3 = retrieving(img=crypt_values,real_img=out2,beta=0.4)
    out4,err4 = retrieving(img=crypt_values,real_img=out3,beta=0.1)
    #error value checking ...
    return [out4,err4]

# Getting Forier module ###

def prep_fft(raw_image):
    if not isinstance(raw_image, np.ndarray):
        image_array = np.array(raw_image)
    else:
        image_array = raw_image
    fft_result = cp.abs(FT(image_array))
    #print(cp.max(fft_result))
    abs_fft = cp.asnumpy(fft_result/cp.max(fft_result)*2**24-1)
    return abs_fft

#IMG_PATH = ['crypt.jpg']
IMG_PATH='output_image.jpg'
#crypt = Image.open(IMG_PATH[0]).convert('RGB')
crypt = Image.open(IMG_PATH).convert('L')
# Image preparation
#crypt_values = img_prep(crypt)

crypt_values=prep_fft(crypt)

### Profiling ###
'''
def profile_func():
    error = 1.
    if use_secrets:
        random_field = generate_secret_random_complex_field()
    else:
        random_field = generate_random_complex_field()
    field_1, error = retr_block(np.abs(random_field))
    field_1, error = retr_block(field_1)
    show_2_images(np.abs(field_1), np.angle(field_1))
    print(error)

cProfile.run('profile_func()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumtime').print_stats(20)
'''
error=1.
for i in range(100):
#while (error>0.0765):
  if use_secrets:
      random_field = generate_secret_random_complex_field()
  else:
      random_field = generate_random_complex_field()
  field_1, error = retr_block(inp=random_field)
  for i in range (25):
    field_1, error = retr_block(inp=np.abs(field_1))
  print(error)
  show_2_images(np.abs(field_1),np.angle(field_1))