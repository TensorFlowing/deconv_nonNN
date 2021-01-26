import numpy as np 

def load_raw(filepath, w=3280, h=2464): 
    raw = np.fromfile(filepath, dtype=np.uint16, count=w*h) 
    raw = np.reshape(raw, (h, w)) 
    return raw 

def raw_to_numpy_uint(filepath, w=3280, h=2464): 
    raw = load_raw(filepath, w, h) 
    c0 = raw[::2, ::2] 
    c1 = raw[::2, 1::2] 
    c2 = raw[1::2, ::2] 
    c3 = raw[1::2, 1::2] 
    img = np.stack([c0, c1, c2, c3], axis=-1) 
    return img 

def raw_to_numpy_float(filepath, w=3280, h=2464, bl=63, wl=1023): 
    raw = load_raw(filepath, w, h) 
    c0 = raw[::2, ::2] 
    c1 = raw[::2, 1::2] 
    c2 = raw[1::2, ::2] 
    c3 = raw[1::2, 1::2] 
    img = np.stack([c0, c1, c2, c3], axis=-1) 
    img = img.astype(np.float) 
    img = (img - bl) / (wl - bl) 
    img = np.clip(img, 0, 1) 
    return img 

def uint_to_float(img, bl=63, wl=1023): 
    img = img.astype(np.float) 
    img = (img - bl) / (wl - bl) 
    img = np.clip(img, 0, 1) 
    return img 

def raw_to_rgb(img, neutral=None, gamma=1.0, channels=4):     
    if channels == 4: 
        img = np.stack([img[:, :, 0], (img[:, :, 1] + img[:, :, 2]) / 2, img[:, :, 3]], axis=-1) 
        if neutral is not None: 
            neutral = np.reshape(np.array(neutral), (1, 1, 3)) 
            img = img / neutral 
    else: 
        img = np.squeeze(img) 
    img = np.clip(img, 0, 1) 
    img = np.power(img, 1/gamma) 
    return img 

def to_4ch(img): 
    return np.stack([img[:, :, 0], img[:, :, 1], img[:, :, 1], img[:, :, 2]], axis=-1) 