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
import matplotlib.animation as animation

def show_2_images(data1, data2,title=" ", title1="Module", title2="Phase", cmap1='bone', cmap2='bone'):
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray) and data1.shape == data2.shape:
        plt.figure(figsize=(12, 6))
        plt.suptitle(title)
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

### IMAGE PREPARATION ###

image_path = "1.tif"
crypt = Image.open(image_path)

Width, Height = crypt.size

# Double precision
crypt_values = np.array(crypt, dtype=np.float64)
noise=34.0
crypt_values=crypt_values-noise
crypt_values = np.where(crypt_values <= 0.0, 0.0, crypt_values)

crypt_values=np.sqrt(crypt_values)

# Check for correct reading
print("Min value:", np.min(crypt_values))
print("Max value:", np.max(crypt_values))

# As in Wolfram document
#crypt_values = np.flipud(crypt_values)

#######################################


### MASK CREATION ###

#mask_width = 384  
#mask_height = 367

mask_width = 367  
mask_height = 384


horizontal_offset = 17  # смещение вправо
Mask = np.zeros((Height, Width), dtype=np.float64)

center_x = Width // 2
center_y = Height // 2

start_x = center_x - (mask_width // 2) - horizontal_offset
start_y = center_y - (mask_height // 2)

Mask[start_y:start_y + mask_height, start_x:start_x + mask_width] = 1.0
Antimask = 1.0 - Mask

####################


### EXPERIMENTAL VALUES AND VIRTUAL LENS ###

Z=60*10/0.006 #cm->mm
Lambda=633*1e-6/0.006 #nm->mm
K=2*np.pi/Lambda
dx=1. #0.006 #mm

crypt_values=crypt_values*Z/(2*np.pi*K)

x_grid = (np.arange(Width) - (Width - 1) // 2-horizontal_offset) * dx
y_grid = (np.arange(Height) - (Height - 1) // 2) * dx

X, Y = np.meshgrid(x_grid, y_grid)
r_squared = X**2 + Y**2  
r_squared=cp.asarray(r_squared)
Exp_ph=cp.exp(1j*K*r_squared/(2*Z))

Exp_ph_cc=cp.exp(-1j*K*r_squared/(2*Z))
#show_2_images(np.abs(np.fft.ifft2(crypt_values)),np.angle(np.fft.ifft2(crypt_values)))
################################


### RANDOM FIELD ###

@njit(parallel=True)
def generate_random_complex_field(width=Width, height=Height):
    rand_field = np.empty((height, width), dtype=np.complex128)
    for i in prange(height):
        for j in range(width):
            amplitude = np.random.randint(0, 1023)
            phase = np.random.random() * 2 * np.pi
            rand_field[i, j] = amplitude * np.exp(1j * phase)
    return rand_field

def generate_secret_random_complex_field(width=Width, height=Height):
    rand_field = np.empty((height, width), dtype=np.complex128)
    amplitudes = np.empty((height, width))
    phases = np.empty((height, width))
    for i in prange(height):
        for j in range(width):
            amplitudes[i,j] = secrets.randbelow(1023)
            phases[i,j] = 0.0#secrets.randbits(64) / 2**64 * 2 * np.pi
    rand_field = amplitudes * np.exp(1j * phases)
    return rand_field

######################

### FFT&IFFT ###

def FT(data):
    #data_gpu = cp.asarray(data)
    result_gpu = cp.fft.fft2(data)
    return result_gpu
def IFT(data):
  data_gpu = cp.asarray(data)
  result_gpu = cp.fft.ifft2(data_gpu)
  return result_gpu
################

### PROJECTOR ###
def steps(X_inp, X_source):
  X_inp=cp.asarray(X_inp)*Exp_ph
  #X_inp=cp.asarray(X_inp)
  X_inp=IFT(cp.asarray(X_source) * cp.exp(1j * cp.angle(FT(X_inp))))
  X_inp=X_inp*Exp_ph_cc
  return cp.asnumpy(X_inp)

def apply_mask2field(mask,field):
  mask=cp.asarray(mask)
  field=cp.asarray(field)
  mod=(cp.abs(field)*mask)
  phase=(cp.angle(field)*mask)
  return(cp.asnumpy(mod*np.exp(1j*phase)))

#################


### ER ###
def ER(N_iter: int, target, source):
    A = target  # 0th iteration --- random field distribution or the previous result
    for i in range(N_iter):
        D = steps(A, source)
        A=apply_mask2field(Mask,D)
    D_norm = cp.linalg.norm(D)
    A_norm = cp.linalg.norm(A)
    Error = cp.sqrt(D_norm**2 - A_norm**2) / D_norm
    return [A, Error]
##########

### HIO ###
def HIO(N_iter: int, beta: float, Target, Source):
    A = Target  # 0th iteration --- random field distribution or the previous result
    for i in range(N_iter):
        D = steps(A, Source)
        A = apply_mask2field(Mask,D) + apply_mask2field(Antimask, (A - beta * D))  # r-domain
    return A
###########

#from the task (30 HIO +10 ER)
def retrieving(img, real_img, beta):
    #real_img is the r-domain field for the HIO's 0th iteration, img --- "true" reciporal magnitude
    c_hio = HIO(N_iter=30, beta=beta, Target=real_img, Source=img)
    #got a c_hio field (in r-domain), now put it into 0th iteration for ER
    c_er,err = ER(10, target=c_hio, source=img)
    return [c_er,err]

############################################################
# This algorithm is taken from the work:                   #
# Artyukov, I.A., Vinogradov, A.V., Gorbunkov, M.V. et al. #
# Virtual Lens Method for Near-Field Phase Retrieval.      #
# Bull. Lebedev Phys. Inst. 50, 414–419 (2023).            #
# https://doi.org/10.3103/S1068335623100020                #
############################################################

def retr_block(inp):
    out1,err1 = retrieving(img=crypt_values,real_img=inp, beta=1.)
    out2,err2 = retrieving(img=crypt_values,real_img=out1,beta=0.7)
    out3,err3 = retrieving(img=crypt_values,real_img=out2,beta=0.4)
    out4,err4 = retrieving(img=crypt_values,real_img=out3,beta=0.1)
    return [out4,err4]

for i in range(100):
  if use_secrets:
      random_field = generate_secret_random_complex_field()
  else:
      random_field = generate_random_complex_field()
  #show_2_images(np.abs(random_field*cp.asnumpy(Exp_ph)),np.angle(random_field*cp.asnumpy(Exp_ph)))
  field_1, error = retr_block(inp=random_field)
  for i in range (20):
    field_1, error = retr_block(inp=np.abs(field_1))
    #print(error)
  #field_1, error = ER(200,target=np.abs(field_1),source=crypt_values)
    show_2_images(data1=np.abs(field_1),data2=np.angle(field_1),title=f"Error:{error:.4f}")

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
#amimation
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

error=1.

if use_secrets:
  random_field = generate_secret_random_complex_field()
else:
  random_field = generate_random_complex_field()
field_1, error = retr_block(inp=random_field)
def update(frame):
    global field_1, error
    field_1, error = retr_block(inp=np.abs(field_1))

    ax1.clear()
    ax2.clear()


    ax1.imshow(np.abs(field_1), cmap='bone')
    ax1.set_title("Module")
    ax2.imshow(np.angle(field_1), cmap='bone')
    ax2.set_title("Phase")


    fig.suptitle(f"Error: {error:.4f}")

    return ax1, ax2

ani = animation.FuncAnimation(fig, update, frames=150, interval=200, blit=False)

#plt.show()

ani.save('ani.mp4', writer='ffmpeg')
'''

