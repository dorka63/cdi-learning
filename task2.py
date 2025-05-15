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
from functools import wraps
from dataclasses import dataclass
from typing import Tuple, Dict, Any,List
import cupyx.scipy.ndimage as cndimage
import imageio

def validate(*arg_positions):
    """Error checking"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            for pos in arg_positions:
                if pos < len(args) and not isinstance(args[pos], np.ndarray):
                    raise ValueError(f"{pos} must be numpy")
            return f(*args, **kwargs)
        return wrapper
    return decorator

def plot_image(data, ax=None, **kwargs):
    """
    Basic function for image drawing.

    Parameters:
        data: np.ndarray
        ax: matplotlib axis (if None, new axis is created)

    """
    if ax is None:
        ax = plt.gca()

    imshow_kwargs = {
        'cmap': kwargs.get('cmap', 'grey'),
        'aspect': kwargs.get('aspect', 'auto'),
        'origin': kwargs.get('origin', 'lower')
    }

    if data.ndim == 3 and data.shape[2] == 3:
        imshow_kwargs.pop('cmap', None)

    img = ax.imshow(data, **imshow_kwargs)
    ax.set_title(kwargs.get('title', ''))

    if kwargs.get('colorbar', True):
        plt.colorbar(img, ax=ax)

    return ax

@validate(0)
def show_1_image(data, **kwargs):
    """Plots 1 image using kwargs"""
    plt.figure(figsize=kwargs.get('figsize', (6, 6)))
    plot_image(data, **kwargs)
    plt.tight_layout()
    plt.show()

@validate(0, 1)
def show_2_images(data1, data2, **kwargs):
    """Plots 2 images using kwargs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=kwargs.get('figsize', (12, 6)))

    plot_image(data1, ax=ax1,
              title=kwargs.get('title1', 'Module'),
              cmap=kwargs.get('cmap1', 'grey'),
              **kwargs)

    plot_image(data2, ax=ax2,
              title=kwargs.get('title2', 'Phase'),
              cmap=kwargs.get('cmap2', 'grey'),
              **kwargs)

    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])

    plt.tight_layout()
    plt.show()

### IMAGE PREPARATION ###

image_path = "1.tif"
crypt = Image.open(image_path)

# Width must be 1030 height 1288
crypt = crypt.rotate(90, expand=True)
crypt_array = np.array(crypt)
crypt_array = np.flipud(crypt_array)
print("Type:", crypt_array.dtype)  #uint16

noise=34
# if value > noise, write value-nose, else write 0
crypt_values = np.where(crypt_array < noise, 0, crypt_array-noise)

#square root of intensity
crypt_values=np.sqrt(crypt_values)
crypt_values=crypt_values.astype(np.complex128)

# Check for correct noise removal
print("Min value:", np.min(crypt_values))
print("Max value:", np.max(crypt_values))

# Size check
Height, Width= crypt_values.shape
print("Width: ", Width, "Height: ", Height) #Width: 1030 Height: 1288

#######################################

### EXPERIMENTAL VALUES ###

Z=60*10*1000 #um
Lambda=632.8/1e3 #nm->um
K=2.*np.pi/Lambda

mask_width_um = 2.3 * 1000  # 2300 um
mask_height_um = 2.2 * 1000  # 2200 um

# detector size (в um)
det_width_um = 0.618 * 10 * 1000  # 6180 um
det_height_um = 0.7728 * 10 * 1000  # 7728 um

# grid step (um/px)
dx = det_width_um / Width
dy = det_height_um / Height

#######################################

### MASK CREATION ###

mask_width = int(mask_width_um /dx) #px
mask_height = int(mask_height_um /dy) #px

offset = 17 #px
Mask = np.zeros((Height, Width), dtype=np.float64)

center_x = Width // 2
center_y = Height // 2-offset

start_x = center_x - (mask_width // 2)
start_y = center_y - (mask_height // 2)

Mask[start_y:start_y + mask_height, start_x:start_x + mask_width] = 1.0
Antimask = 1.0 - Mask

#######################################

### VIRTUAL LENS ###

#Formula (9)
#crypt_values=crypt_values*Z/(2.*np.pi*K)
#show_2_images(np.abs(crypt_values),np.angle(crypt_values))

x_grid = (np.arange(Width) - (Width // 2)) * dx
y_grid = (np.arange(Height) - (Height // 2)+offset) * dy
X, Y = np.meshgrid(x_grid, y_grid)
r_squared = X**2 + Y**2
r_squared=cp.asarray(r_squared)

Exp_ph=cp.exp(1j*K*r_squared/(2*Z))
Exp_ph_c=cp.exp(-1j*K*r_squared/(2*Z))


#show_2_images(cp.asnumpy(cp.angle(Exp_ph_c)),cp.asnumpy(cp.angle(Exp_ph)))

def affine_scale(img, kx, ky,d=offset):
    h, w = img.shape
    # Матрица масштабирования (учитывает центр изображения)
    matrix = cp.array([[ky, 0], [0, kx]])
    offset = cp.array([h, w]) * (1 - cp.array([ky, kx])) // 2-cp.array([d,0])
    return cndimage.affine_transform(img, matrix, offset=offset,order=5, output_shape=(h, w))

A_x, A_y = (Lambda*Z)/(det_width_um*dx),(Lambda*Z)/(det_height_um*dy)
crypt_values=affine_scale(cp.asarray(crypt_values),A_x,A_y)
################################

### RANDOM FIELD ###

def generate_random_complex_field(height=Height,width=Width):
    #height,width=Size
    rand_field = np.empty((height, width), dtype=np.complex64)
    amplitudes = np.empty((height, width))
    phases = np.empty((height, width))
    for i in prange(height):
        for j in range(width):
            amplitudes[i,j] = secrets.randbits(32) / 2**32
            phases[i,j] = secrets.randbits(32) / 2**32 * 2 * np.pi
    rand_field = amplitudes * np.exp(1j * phases)
    return rand_field

### FFT&IFFT ###

def FT(data):
    data_gpu = cp.asarray(data)
    data_gpu = cp.fft.ifftshift(data_gpu)
    result_gpu = cp.fft.fft2(data_gpu)
    result_gpu = cp.fft.fftshift(result_gpu)
    return result_gpu

def IFT(data):
    data_gpu = cp.asarray(data)
    data_gpu = cp.fft.ifftshift(data_gpu)
    result_gpu = cp.fft.ifft2(data_gpu)
    result_gpu = cp.fft.fftshift(result_gpu)
    return result_gpu

#######################################


def prop(field, z):
    field_1 = FT(field)
    field_1 = cp.fft.ifftshift(field_1)
    n, m = field_1.shape
    fx = cp.fft.fftfreq(n)
    fy = cp.fft.fftfreq(m)
    px, py = cp.meshgrid(fx, fy, indexing='ij')
    p_squared = px**2 + py**2
    field_1 = field_1 * cp.exp(1j * z * cp.sqrt(K**2 - p_squared))
    field_1 = cp.fft.ifftshift(field_1)
    field_1=IFT(field_1)
    return field_1

crypt_values=crypt_values*Z/(2.*np.pi*K)
#show_2_images(np.abs(cp.asnumpy(prop_det)),np.angle(cp.asnumpy(prop_det)))

### PROJECTOR ###

def steps(Detector, Num_field,virt_lens:bool=True):
  X_1=cp.asarray(Num_field)
  if virt_lens:
    X_2=FT(X_1*Exp_ph)
  else:
    X_2=prop(X_1,Z)
  #X_2=X_2/cp.max(X_2)
  X_3=cp.asarray(Detector)*X_2/cp.abs(X_2)
  if virt_lens:
    X_4=IFT(X_3)/Exp_ph
  else:
    X_4=prop(X_3,-Z)
  X_out=X_4#/cp.max(X_4)
  return X_out

def apply_mask2field(msk,fld):
  mask=cp.asarray(msk)
  field=cp.asarray(fld)
  return (mask*field)

#######################################

### ER ###
def ER(N_iter: int, field, detector):
    A =field  # 0th iteration --- random field distribution or the previous result
    for i in range(N_iter):
        D = steps(Detector=detector,Num_field=A)
        A=apply_mask2field(msk=Mask,fld=D)
    D_norm = cp.linalg.norm(D)
    A_norm = cp.linalg.norm(A)
    Error = cp.sqrt(D_norm**2 - A_norm**2) / D_norm
    return [A, Error]
##########

### HIO ###
def HIO(N_iter: int, beta: float, field, detector):
    A = cp.asarray(field)  # 0th iteration --- random field distribution or the previous result
    for i in range(N_iter):
        D = steps(Detector=detector,Num_field=A)
        A = apply_mask2field(msk=Mask,fld=D) + apply_mask2field(msk=Antimask, fld=(A - beta * D))  # r-domain
    return A
###########

def retrieving(DET, FLD, beta,N):
    # FLD is the r-domain field for the HIO's 0th iteration, DET --- detector's magnitude
    c_hio = HIO(N_iter=30, beta=beta, field=FLD, detector=DET)
    # c_hio field (in r-domain), now put it into 0th iteration for ER
    c_er,err = ER(N_iter=N, field=c_hio, detector=DET)
    return [c_er,err]


def retr_block(input_field,det=crypt_values,N=10):
    out1,err1 = retrieving(DET=det,FLD=input_field, beta=1.,N=N)
    out2,err2 = retrieving(DET=det,FLD=cp.abs(out1),beta=0.7,N=N)
    out3,err3 = retrieving(DET=det,FLD=cp.abs(out2),beta=0.4,N=N)
    out4,err4 = retrieving(DET=det,FLD=cp.abs(out3),beta=0.1,N=N)
    return [out4,err4]

error=1.0
while error>=0.054:
  random_field = generate_random_complex_field()

  field_1, error = retr_block(np.abs(random_field*Mask))
  for i in range (10):
    field_1, error = retr_block(field_1*cp.asarray(Mask),N=100)
    show_2_images(data1=np.abs(cp.asnumpy(field_1)),data2=np.angle(cp.asnumpy(field_1)),suptitle=f"Error:{error:.4f}")
  field_1, error = retr_block(field_1,N=200)
  show_2_images(data1=np.abs(cp.asnumpy(field_1)),data2=np.angle(cp.asnumpy(field_1)),suptitle=f"Error:{error:.4f}")
