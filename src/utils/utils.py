import os
import numpy as np
from PIL import Image

EMOTION_LABELS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

def load_images_from_directory(base_dir, image_size=(48, 48)):
    X = []
    y = []

    for label_name, label_id in EMOTION_LABELS.items():
        folder = os.path.join(base_dir, label_name)
        if not os.path.exists(folder):
            print(f"No folder: {folder}")
            continue

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                img = Image.open(file_path).convert("L")  # グレースケール
                img = img.resize(image_size)
                img_array = np.asarray(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)  # (1, H, W)
                X.append(img_array)
                y.append(label_id)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    return X, y

def load_fer2013(data_root="../", image_size=(48, 48)):
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    x_train, y_train = load_images_from_directory(train_dir, image_size)
    x_test, y_test = load_images_from_directory(test_dir, image_size)

    return x_train, y_train, x_test, y_test

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    filter_h : フィルタの高さ
    filter_w : フィルタの幅     
                                        
    Returns
    -------
    col : 変換後のデータ（形状は(N*out_h*out_w
    , filter_h*filter_w*C)）
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def flatten(x):
    """
    入力: (N, C, H, W)
    出力: (N, C*H*W)
    """
    return x.reshape(x.shape[0], -1)

def unflatten(x, shape):
    return x.reshape(shape)


def he_init(shape):
    if len(shape) == 4:
        fan_in = shape[1] * shape[2] * shape[3]
    elif len(shape) == 2:
        fan_in = shape[0]
    else:
        raise ValueError("Unsupported shape")
    return np.random.randn(*shape) * np.sqrt(2. / fan_in)


def init_lenet_params():
    params = {}
    # Conv1: in=1, out=6, kernel=5x5
    params['W1'] = he_init((6, 1, 5, 5))
    params['b1'] = np.zeros(6)
    
    # Conv2: in=6, out=16, kernel=5x5
    params['W2'] = he_init((16, 6, 5, 5))
    params['b2'] = np.zeros(16)
    
    # FC1: input=16*5*5=400 → 84
    params['W3'] = he_init((1296, 84))
    params['b3'] = np.zeros(84)
    
    # FC2: 84 → 7
    params['W4'] = he_init((84, 7))
    params['b4'] = np.zeros(7)

    return params

def init_vgg_params():
    params = {}
    params['W1'] = he_init((64, 1, 3, 3))
    params['b1'] = np.zeros(64)
    params['W2'] = he_init((64, 64, 3, 3))
    params['b2'] = np.zeros(64)

    params['W3'] = he_init((128, 64, 3, 3))
    params['b3'] = np.zeros(128)
    params['W4'] = he_init((128, 128, 3, 3))
    params['b4'] = np.zeros(128)
    
    params['W5'] = he_init((18432, 256))
    params['b5'] = np.zeros(256)
    params['W6'] = he_init((256, 7))
    params['b6'] = np.zeros(7)
    return params

def init_resnet_params():
    params = {}
    params['W1'] = he_init((64, 1, 3, 3))
    params['b1'] = np.zeros(64)
    params['W2'] = he_init((64, 64, 3, 3))
    params['b2'] = np.zeros(64)

    params['W3'] = he_init((64, 64, 3, 3))
    params['b3'] = np.zeros(64)
    params['W4'] = he_init((64, 64, 3, 3))
    params['b4'] = np.zeros(64)

    params['W5'] = he_init((9216, 128))
    params['b5'] = np.zeros(128)
    params['W6'] = he_init((128, 7))
    params['b6'] = np.zeros(7)
    return params
