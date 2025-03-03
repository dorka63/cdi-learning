import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit,prange
from multiprocessing import Pool
import pyfftw
import cProfile
import pstats

try:
    import secrets 
    use_secrets = True
except ImportError:
    import np.random
    use_secrets = False


def show_2_images(data1, data2, title1="Modulo", title2="Phase", cmap1=None, cmap2=None):
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

#center_square_start = (555 // 2) - (555 // 4)
#center_square_end = (555 // 2) + (555 // 4)
center_square_start = 134
center_square_end = 420

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
            amplitudes[i,j] = secrets.randbelow(2**24 - 1)
            phases[i,j] = secrets.randbits(32) / 2**32 * 2 * np.pi
    rand_field = amplitudes * np.exp(1j * phases)
    return rand_field

# Настройка pyFFTW
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(5)

pyfftw.config.NUM_THREADS = 8
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

def FT(data):
    return pyfftw.interfaces.numpy_fft.fft2(data)

def IFT(data):
    return pyfftw.interfaces.numpy_fft.ifft2(data)

def steps(X_inp, X_source):
    return IFT(X_source * np.exp(1j * np.angle(FT(X_inp))))


@njit
def error_calc(A,B):
    B_norm = np.linalg.norm(B)
    A_norm = np.linalg.norm(A)
    return np.sqrt(B_norm**2 - A_norm**2) / B_norm

#Error Reduction (ER)

def ER(N_iter,  target, source):
    A = target  # 0th iteration
    for i in range(N_iter):
        D = steps(A, source)
        A = D * mask  # r-domain
    Error = error_calc(A,D)
    return [A, Error]

#Hybrid Input-Output (HIO)

def HIO(N_iter, beta, Target, Source):
    A = Target  # 0th iteration
    for i in range(N_iter):
        D = steps(A, Source)
        A = mask * D + antimask * (A - beta * D)  # r-domain
    return A

#from the task (30 HIO +10 ER)
def retrieving(img, real_img, beta):
    #real_img is the r-domain field for the HIO's 0th iteration, img --- "true" reciporal magnitude
    c_hio = HIO(N_iter=30, beta=beta, Target=real_img, Source=img)
    #got a c_hio field (in r-domain), now put it into 0th iteration for ER
    c_er,err = ER(10, target=c_hio, source=img)
    #c_er = ER(10, target=c_hio, source=img)
    return [c_er,err]

# This algorithm is taken from the work:
# Artyukov, I.A., Vinogradov, A.V., Gorbunkov, M.V. et al.
# Virtual Lens Method for Near-Field Phase Retrieval. 
# Bull. Lebedev Phys. Inst. 50, 414–419 (2023)
# https://doi.org/10.3103/S1068335623100020

def retr_block(inp):
    out1,err1 = retrieving(img=crypt_values,real_img=inp, beta=1.)
    out2,err2 = retrieving(img=crypt_values,real_img=out1,beta=0.7)
    out3,err3 = retrieving(img=crypt_values,real_img=out2,beta=0.4)
    out4,err4 = retrieving(img=crypt_values,real_img=out3,beta=0.1)
    #error value checking ...
    return [out4,err4]


IMG_PATH = ['crypt.jpg']
crypt = Image.open(IMG_PATH[0]).convert('RGB')

# Image preparation
crypt_values = img_prep(crypt)

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

def process_image(_):
    if use_secrets:
        random_field = generate_secret_random_complex_field()
    else:
        random_field = generate_random_complex_field()
    field_1, error = retr_block(np.abs(random_field))
    for _ in range(10):
        field_1, error = retr_block(field_1)
    print(error)
    return field_1

if __name__ == "__main__":
    with Pool(processes=4) as pool:  
        results = pool.map(process_image, range(10))
    for field_1 in results:
        show_2_images(np.abs(field_1), np.angle(field_1))
