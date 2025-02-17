import numpy as np

def create_stripe_mask(size=555, unit_stripe_height=16, zero_stripe_height=4, num_stripes=13):
    mask = np.zeros((size, size), dtype=np.uint8)

    center = size // 2
    half_size = size // 4-1

    total_stripe_height = unit_stripe_height+ zero_stripe_height

    for i in range(num_stripes):
        top = center - half_size + i * total_stripe_height
        mask[top:top + unit_stripe_height, :] = 1  # Заполняем 1
        mask[top+ unit_stripe_height:top + total_stripe_height, :] = 0
    return mask
