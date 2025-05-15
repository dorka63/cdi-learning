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

@dataclass
class ExperimentalConf:
    z_um: float = 6e5
    lambda_um: float = 0.6328
    noise_threshold: int = 34
    detector_width_um: float = 0.618 * 1e4
    detector_height_um: float = 0.7728 * 1e4
    mask_width_um: float = 2.2 * 1e3
    mask_height_um: float = 2.3 * 1e3
    mask_x_offset_um: float = 0.0
    mask_y_offset_um: float = 6*17
    rotation_angle: int = 90
    image_path: str = '1.tif'

    _cached_properties: Dict[str, Any] = None

    def __post_init__(self):
        self._cached_properties = {}

    def _get_or_compute(self, key: str, compute_func):
        if key not in self._cached_properties:
            self._cached_properties[key] = compute_func()
        return self._cached_properties[key]

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._get_or_compute('image_size', lambda: (
            (lambda img: (img.width,img.height) if self.rotation_angle in (90, 270) else (img.height,img.width))(
                Image.open(self.image_path)
            )))

    @property
    def pixel_size(self) -> Tuple[float, float]:
        """Returns (dx, dy) in um/px"""
        return self._get_or_compute('pixel_size', lambda: (
            self.detector_height_um / self.image_size[0],
            self.detector_width_um / self.image_size[1]
        ))

    @property
    def mask_offset(self) -> Tuple[int, int]:
        """Returns (offset_x_px, offset_y_px) in px"""
        dy, dx = self.pixel_size
        return (
            int(self.mask_x_offset_um / dx),
            int(self.mask_y_offset_um / dy)
        )

    @property
    def scale_factors(self) -> Tuple[float, float]:
        """Scaling for affine_transform"""
        dx, dy = self.pixel_size
        return (
            (self.lambda_um * self.z_um) / (self.detector_width_um * dx),
            (self.lambda_um * self.z_um) / (self.detector_height_um * dy)
        )

class ImageProcessor:
    def __init__(self, config: ExperimentalConf):
        self.config = config
        self.k = 2 * np.pi / self.config.lambda_um

    def load_and_preprocess(self,scale_flag:bool=True) -> cp.ndarray:
        img = np.flipud(np.array(
            Image.open(self.config.image_path).rotate(self.config.rotation_angle, expand=True)
        ))
        clean = np.where(img < self.config.noise_threshold, 0, img - self.config.noise_threshold)
        if scale_flag:
          return cp.asarray(np.sqrt(clean) * self.config.z_um / (2 * np.pi * self.k))
        else:
          return cp.asarray(np.sqrt(clean))


    def generate_random_complex_field(self) ->cp.ndarray:
        height,width=self.config.image_size
        rand_field = np.empty((height, width), dtype=np.complex64)
        amplitudes = np.empty((height, width))
        phases = np.empty((height, width))
        for i in prange(height):
            for j in range(width):
                amplitudes[i,j] = secrets.randbits(32) / 2**32
                phases[i,j] = secrets.randbits(32) / 2**32 * 2 * np.pi
        rand_field = amplitudes * np.exp(1j * phases)
        return cp.asarray(rand_field)


    def create_mask(self) -> cp.ndarray:
        height, width = self.config.image_size
        dx, dy = self.config.pixel_size
        offset_px, offset_py = self.config.mask_offset

        mask = cp.zeros((height, width), dtype=cp.float64)
        mask_width_px = int(self.config.mask_width_um / dx)
        mask_height_px = int(self.config.mask_height_um / dy)

        start_x = (width // 2 - offset_px) - (mask_width_px // 2)
        start_y = (height // 2 - offset_py) - (mask_height_px // 2)

        mask[start_y:start_y+mask_height_px, start_x:start_x+mask_width_px] = 1.0
        return mask

    def create_phase_exp(self) ->cp.ndarray:
        height, width = self.config.image_size
        dx, dy = self.config.pixel_size
        offset_px, offset_py = self.config.mask_offset

        x = (cp.arange(width) - (width//2 - offset_px)) * dx
        y = (cp.arange(height) - (height//2 - offset_py)) * dy
        X, Y = cp.meshgrid(x, y)
        r_sq = X**2 + Y**2

        phase = 1j * self.k * r_sq / (2 * self.config.z_um)
        return cp.exp(phase)

    def apply_affine_scale(self) -> cp.ndarray:
      kx, ky = self.config.scale_factors
      img = self.load_and_preprocess(scale_flag=True)
      h, w = img.shape

      offset_px, offset_py = self.config.mask_offset

      matrix = cp.array([[ky, 0], [0, kx]])
      offset = cp.array([h, w]) * (1 - cp.array([ky, kx])) // 2-cp.array([offset_py,offset_px])

      return cndimage.affine_transform(img,matrix,offset=offset,order=5,output_shape=(h, w))

class PhaseRetriever(ImageProcessor):
    def __init__(self, config: ExperimentalConf):
        super().__init__(config)
        self._phase_exp = self.create_phase_exp()
        self.mask = self.create_mask()
        self.flag_prop=False

    @property
    def phase_exp(self) -> cp.ndarray:
        return self._phase_exp

    def ft(self, data: np.ndarray) -> cp.ndarray:
        data_gpu = cp.asarray(data, dtype=np.complex64)
        return cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(data_gpu)))

    def ift(self, data: np.ndarray) -> cp.ndarray:
        data_gpu = cp.asarray(data, dtype=np.complex64)
        return cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(data_gpu)))


    def prop(self, field:np.ndarray,Z:float) ->cp.ndarray:
        field_1 = self.ft(field)
        field_1 = cp.fft.ifftshift(field_1)
        n, m = field_1.shape
        fx = cp.fft.fftfreq(n)
        fy = cp.fft.fftfreq(m)
        px, py = cp.meshgrid(fx, fy, indexing='ij')
        p_squared = px**2 + py**2  # Squared magnitude of spatial frequencies
        field_1 = field_1 * cp.exp(1j * Z * cp.sqrt(self.k**2 - p_squared))
        field_1 = cp.fft.ifftshift(field_1)
        field_1=self.ift(field_1)
        return field_1

    def step(self, field: cp.ndarray, detector: np.ndarray) -> cp.ndarray:
        X_1=cp.asarray(field)
        flag_prop=self.flag_prop
        if flag_prop:
          X_2=self.prop(cp.asnumpy(X_1),Z=self.config.z_um)
        else:
          X_2=self.ft(X_1*self.phase_exp)
        #X_2=X_2/cp.max(X_2)
        X_3=cp.asarray(detector)*X_2/cp.abs(X_2)
        if flag_prop:
          X_4=self.prop(X_3,Z=-self.config.z_um)
        else:
          X_4=self.ift(X_3)/self.phase_exp
        X_out=X_4#/cp.max(X_4)
        return X_out

    def ER(self,n_iter: int, field: cp.ndarray, detector: np.ndarray) -> Tuple[cp.ndarray, float]:
        """Error Reduction"""
        field_1 = cp.asarray(field)

        for _ in range(n_iter):
            d = self.step(field_1,detector)
            field_1 = self.mask*d

        d_norm = cp.linalg.norm(d)
        a_norm = cp.linalg.norm(field_1)
        error = cp.sqrt(d_norm**2 - a_norm**2) / d_norm

        return cp.asnumpy(field_1), float(error)


    def HIO(self, n_iter: int, field: cp.ndarray, detector: np.ndarray, beta: float) -> cp.ndarray:
        """Hybrid Input-Output"""
        field_1 = cp.asarray(field)

        for _ in range(n_iter):
            d = self.step(field_1,detector)
            field_1 = self.mask*d +(field_1 - beta * d)*(1.0-self.mask)

        return cp.asnumpy(field_1)

    def retrieving(self,DET:np.ndarray, FLD:np.ndarray, beta:float,hio_iters: int = 30, er_iters: int = 10)->Tuple[np.ndarray, float]:
        # FLD is the r-domain field for the HIO's 0th iteration, DET --- detector's magnitude
        c_hio = self.HIO(n_iter=hio_iters, beta=beta, field=FLD, detector=DET)
        # c_hio field (in r-domain), now put it into 0th iteration for ER
        c_er,err = self.ER(n_iter=er_iters, field=c_hio, detector=DET)
        return [c_er,err]

    def retr_block(self, detector: np.ndarray, initial_field: np.ndarray,
                hio_iters: int = 30, er_iters: int = 10,betas:List=[0.7,0.4,0.1]) -> Tuple[np.ndarray, float]:
        """
        Phase retrival cycle

        ############################################################
        # This algorithm is taken from the work:                   #
        # Artyukov, I.A., Vinogradov, A.V., Gorbunkov, M.V. et al. #
        # Virtual Lens Method for Near-Field Phase Retrieval.      #
        # Bull. Lebedev Phys. Inst. 50, 414–419 (2023).            #
        # https://doi.org/10.3103/S1068335623100020                #
        ############################################################

        Params:
            detector: Detector's image [numpy array]
            initial_field: Initial field (firstly, random) [numpy array]
            hio_iters: Number of HIO's iterations for each beta value
            er_iters: Number of ER's iterations for each beta value

        Returns:
            Retrieved field and error
        """
        current_field = cp.asarray(initial_field)
        error = 0.
        current_field,error = self.retrieving(DET=detector,FLD=current_field,beta=1.0)
        for beta in betas:
          current_field,error=self.retrieving(DET=detector,FLD=(current_field),beta=beta)
        return cp.asnumpy(current_field), error


letter_F_config = ExperimentalConf(image_path="1.tif")
F_processor = ImageProcessor(letter_F_config)

#CHECK#
print("Type (must be uint16):", np.asarray(Image.open(letter_F_config.image_path)).dtype)
print("Image size [Height, Width] (px):", F_processor.config.image_size)  #Height 1288, Width 1030
print("Pixel size (um):", F_processor.config.pixel_size)
print("Mask offset (px):", F_processor.config.mask_offset)
print(letter_F_config.scale_factors)

F_mask=F_processor.create_mask()
detector_F_data_wp=F_processor.load_and_preprocess(scale_flag=False)
detector_F_data_vl=F_processor.apply_affine_scale()
show_2_images(cp.asnumpy(detector_F_data_vl),cp.asnumpy(detector_F_data_wp),title1="Scaled image from the detector",title2="Image from the detector")
VL_F=PhaseRetriever(letter_F_config)
WP_F=PhaseRetriever(letter_F_config)
WP_F.flag_prop=True
initial_field = F_mask*F_processor.generate_random_complex_field()

#show_2_images(np.abs(cp.asnumpy(initial_field)),np.angle(cp.asnumpy(initial_field)))

def MAINFUNC(vl:bool=True,show:bool=True):
  if vl:
    retrieved=VL_F
    det_dat=detector_F_data_vl
  else:
    retrieved=WP_F
    det_dat=detector_F_data_wp
  frames = []
  result, errors = retrieved.retr_block(
      detector=det_dat,
      initial_field=np.abs(initial_field))
  for i in range(10):
    result, errors = retrieved.retr_block(
    detector=det_dat,
    initial_field=np.abs(result*cp.asnumpy(F_mask)),er_iters=10)
    if show:
      show_2_images(np.abs(cp.asnumpy(result)),np.angle(cp.asnumpy(result)),suptitle=f"Error:{errors:.4f}")

  result, errors = retrieved.retr_block(
  detector=det_dat,
  initial_field=np.abs(result*cp.asnumpy(F_mask)),er_iters=200)
  if show:
    show_2_images(np.abs(cp.asnumpy(result)),np.angle(cp.asnumpy(result)),suptitle=f"Error:{errors:.4f}")


MAINFUNC(vl=False,show=True)

cProfile.run('MAINFUNC(vl=False,show=False)','profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumtime').print_stats(20)