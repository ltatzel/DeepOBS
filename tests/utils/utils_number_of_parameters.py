"""Simple dictionary of layers per test problem (split per layer)"""

NUMBER_OF_PARAMETERS = {
    "cifar100_3c3d": [
        3 * 64 * 5 * 5,
        64,
        64 * 96 * 3 * 3,
        96,
        96 * 128 * 3 * 3,
        128,
        3 * 3 * 128 * 512,
        512,
        512 * 256,
        256,
        256 * 100,
        100,
    ],
    "cifar100_allcnnc": [
        3 * 96 * 3 * 3,
        96,
        96 * 96 * 3 * 3,
        96,
        96 * 96 * 3 * 3,
        96,
        96 * 192 * 3 * 3,
        192,
        192 * 192 * 3 * 3,
        192,
        192 * 192 * 3 * 3,
        192,
        192 * 192 * 3 * 3,
        192,
        192 * 192 * 1 * 1,
        192,
        192 * 100 * 1 * 1,
        100,
    ],
    "cifar100_vgg16": [
        3 * 64 * 3 * 3,
        64,
        64 * 64 * 3 * 3,
        64,
        64 * 128 * 3 * 3,
        128,
        128 * 128 * 3 * 3,
        128,
        128 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 7 * 7 * 4096,
        4096,
        4096 * 4096,
        4096,
        4096 * 100,
        100,
    ],
    "cifar100_vgg19": [
        3 * 64 * 3 * 3,
        64,
        64 * 64 * 3 * 3,
        64,
        64 * 128 * 3 * 3,
        128,
        128 * 128 * 3 * 3,
        128,
        128 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 7 * 7 * 4096,
        4096,
        4096 * 4096,
        4096,
        4096 * 100,
        100,
    ],
    "cifar100_wrn164": [
        3 * 16 * 3 * 3,
        16,
        16,
        16 * 64 * 1 * 1,
        16 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 128 * 1 * 1,
        64 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 256 * 1 * 1,
        128 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 100,
        100,
    ],
    "cifar100_wrn404": [
        3 * 16 * 3 * 3,
        16,
        16,
        16 * 64 * 1 * 1,
        16 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 128 * 1 * 1,
        64 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 256 * 1 * 1,
        128 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 100,
        100,
    ],
    "cifar10_3c3d": [
        3 * 64 * 5 * 5,
        64,
        64 * 96 * 3 * 3,
        96,
        96 * 128 * 3 * 3,
        128,
        3 * 3 * 128 * 512,
        512,
        512 * 256,
        256,
        256 * 10,
        10,
    ],
    "cifar10_vgg16": [
        3 * 64 * 3 * 3,
        64,
        64 * 64 * 3 * 3,
        64,
        64 * 128 * 3 * 3,
        128,
        128 * 128 * 3 * 3,
        128,
        128 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 7 * 7 * 4096,
        4096,
        4096 * 4096,
        4096,
        4096 * 10,
        10,
    ],
    "cifar10_vgg19": [
        3 * 64 * 3 * 3,
        64,
        64 * 64 * 3 * 3,
        64,
        64 * 128 * 3 * 3,
        128,
        128 * 128 * 3 * 3,
        128,
        128 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 7 * 7 * 4096,
        4096,
        4096 * 4096,
        4096,
        4096 * 10,
        10,
    ],
    "fmnist_2c2d": [
        1 * 32 * 5 * 5,
        32,
        32 * 64 * 5 * 5,
        64,
        7 * 7 * 64 * 1024,
        1024,
        1024 * 10,
        10,
    ],
    "fmnist_logreg": [784 * 10, 10],
    "fmnist_mlp": [
        28 * 28 * 1000,
        1000,
        1000 * 500,
        500,
        500 * 100,
        100,
        100 * 10,
        10,
    ],
    "fmnist_vae": [
        1 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        7 * 7 * 64 * 8,
        8,
        7 * 7 * 64 * 8,
        8,
        8 * 24,
        24,
        24 * 49,
        49,
        1 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        14 * 14 * 64 * 28 * 28,
        28 * 28,
    ],
    "imagenet_inception_v3": [
        3 * 32 * 3 * 3,
        32,
        32,
        32 * 32 * 3 * 3,
        32,
        32,
        32 * 64 * 3 * 3,
        64,
        64,
        64 * 80 * 1 * 1,
        80,
        80,
        80 * 192 * 3 * 3,
        192,
        192,
        192 * 64 * 1 * 1,
        64,
        64,
        192 * 32 * 1 * 1,
        32,
        32,
        192 * 48 * 1 * 1,
        48,
        48,
        48 * 64 * 5 * 5,
        64,
        64,
        192 * 64 * 1 * 1,
        64,
        64,
        64 * 96 * 3 * 3,
        96,
        96,
        96 * 96 * 3 * 3,
        96,
        96,
        (64 + 32 + 64 + 96) * 64 * 1 * 1,
        64,
        64,
        (64 + 32 + 64 + 96) * 64 * 1 * 1,
        64,
        64,
        (64 + 32 + 64 + 96) * 48 * 1 * 1,
        48,
        48,
        48 * 64 * 5 * 5,
        64,
        64,
        (64 + 32 + 64 + 96) * 64 * 1 * 1,
        64,
        64,
        64 * 96 * 3 * 3,
        96,
        96,
        96 * 96 * 3 * 3,
        96,
        96,
        (64 + 64 + 64 + 96) * 64 * 1 * 1,
        64,
        64,
        (64 + 64 + 64 + 96) * 64 * 1 * 1,
        64,
        64,
        (64 + 64 + 64 + 96) * 48 * 1 * 1,
        48,
        48,
        48 * 64 * 5 * 5,
        64,
        64,
        (64 + 64 + 64 + 96) * 64 * 1 * 1,
        64,
        64,
        64 * 96 * 3 * 3,
        96,
        96,
        96 * 96 * 3 * 3,
        96,
        96,
        (64 + 64 + 64 + 96) * 384 * 3 * 3,
        384,
        384,
        (64 + 64 + 64 + 96) * 64 * 1 * 1,
        64,
        64,
        64 * 96 * 3 * 3,
        96,
        96,
        96 * 96 * 3 * 3,
        96,
        96,
        ((64 + 64 + 64 + 96) + 384 + 96) * 192 * 1 * 1,
        192,
        192,
        ((64 + 64 + 64 + 96) + 384 + 96) * 192 * 1 * 1,
        192,
        192,
        ((64 + 64 + 64 + 96) + 384 + 96) * 128 * 1 * 1,
        128,
        128,
        128 * 128 * 1 * 7,
        128,
        128,
        128 * 192 * 7 * 1,
        192,
        192,
        ((64 + 64 + 64 + 96) + 384 + 96) * 128 * 1 * 1,
        128,
        128,
        128 * 128 * 7 * 1,
        128,
        128,
        128 * 128 * 1 * 7,
        128,
        128,
        128 * 128 * 7 * 1,
        128,
        128,
        128 * 192 * 1 * 7,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 160 * 1 * 1,
        160,
        160,
        160 * 160 * 1 * 7,
        160,
        160,
        160 * 192 * 7 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 160 * 1 * 1,
        160,
        160,
        160 * 160 * 7 * 1,
        160,
        160,
        160 * 160 * 1 * 7,
        160,
        160,
        160 * 160 * 7 * 1,
        160,
        160,
        160 * 192 * 1 * 7,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 160 * 1 * 1,
        160,
        160,
        160 * 160 * 1 * 7,
        160,
        160,
        160 * 192 * 7 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 160 * 1 * 1,
        160,
        160,
        160 * 160 * 7 * 1,
        160,
        160,
        160 * 160 * 1 * 7,
        160,
        160,
        160 * 160 * 7 * 1,
        160,
        160,
        160 * 192 * 1 * 7,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        192 * 192 * 1 * 7,
        192,
        192,
        192 * 192 * 7 * 1,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        192 * 192 * 7 * 1,
        192,
        192,
        192 * 192 * 1 * 7,
        192,
        192,
        192 * 192 * 7 * 1,
        192,
        192,
        192 * 192 * 1 * 7,
        192,
        192,
        (192 + 192 + 192 + 192) * 128 * 1 * 1,
        128,
        128,
        128 * 768 * 5 * 5,
        768,
        768,
        768 * 1001,
        1001,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        192 * 192 * 1 * 7,
        192,
        192,
        192 * 192 * 7 * 1,
        192,
        192,
        192 * 192 * 3 * 3,
        192,
        192,
        (192 + 192 + 192 + 192) * 192 * 1 * 1,
        192,
        192,
        192 * 320 * 3 * 3,
        320,
        320,
        (4 * 192 + 192 + 320) * 320 * 1 * 1,
        320,
        320,
        (4 * 192 + 192 + 320) * 384 * 1 * 1,
        384,
        384,
        384 * 384 * 1 * 3,
        384,
        384,
        384 * 384 * 3 * 1,
        384,
        384,
        (4 * 192 + 192 + 320) * 448 * 1 * 1,
        448,
        448,
        448 * 384 * 3 * 3,
        384,
        384,
        384 * 384 * 1 * 3,
        384,
        384,
        384 * 384 * 3 * 1,
        384,
        384,
        (4 * 192 + 192 + 320) * 192,
        192,
        192,
        (320 + 384 * 2 + 384 * 2 + 192) * 320 * 1 * 1,
        320,
        320,
        (320 + 384 * 2 + 384 * 2 + 192) * 384 * 1 * 1,
        384,
        384,
        384 * 384 * 1 * 3,
        384,
        384,
        384 * 384 * 3 * 1,
        384,
        384,
        (320 + 384 * 2 + 384 * 2 + 192) * 448 * 1 * 1,
        448,
        448,
        448 * 384 * 3 * 3,
        384,
        384,
        384 * 384 * 1 * 3,
        384,
        384,
        384 * 384 * 3 * 1,
        384,
        384,
        (320 + 384 * 2 + 384 * 2 + 192) * 192,
        192,
        192,
        2048 * 1001,
        1001,
    ],
    "imagenet_vgg16": [
        3 * 64 * 3 * 3,
        64,
        64 * 64 * 3 * 3,
        64,
        64 * 128 * 3 * 3,
        128,
        128 * 128 * 3 * 3,
        128,
        128 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 7 * 7 * 4096,
        4096,
        4096 * 4096,
        4096,
        4096 * 1001,
        1001,
    ],
    "imagenet_vgg19": [
        3 * 64 * 3 * 3,
        64,
        64 * 64 * 3 * 3,
        64,
        64 * 128 * 3 * 3,
        128,
        128 * 128 * 3 * 3,
        128,
        128 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 7 * 7 * 4096,
        4096,
        4096 * 4096,
        4096,
        4096 * 1001,
        1001,
    ],
    "mnist_2c2d": [
        1 * 32 * 5 * 5,
        32,
        32 * 64 * 5 * 5,
        64,
        7 * 7 * 64 * 1024,
        1024,
        1024 * 10,
        10,
    ],
    "mnist_logreg": [784 * 10, 10],
    "mnist_mlp": [28 * 28 * 1000, 1000, 1000 * 500, 500, 500 * 100, 100, 100 * 10, 10,],
    "mnist_vae": [
        1 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        7 * 7 * 64 * 8,
        8,
        7 * 7 * 64 * 8,
        8,
        8 * 24,
        24,
        24 * 49,
        49,
        1 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        14 * 14 * 64 * 28 * 28,
        28 * 28,
    ],
    "quadratic_deep": [100],
    "svhn_3c3d": [
        3 * 64 * 5 * 5,
        64,
        64 * 96 * 3 * 3,
        96,
        96 * 128 * 3 * 3,
        128,
        3 * 3 * 128 * 512,
        512,
        512 * 256,
        256,
        256 * 10,
        10,
    ],
    "svhn_wrn164": [
        3 * 16 * 3 * 3,
        16,
        16,
        16 * 64 * 1 * 1,
        16 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 128 * 1 * 1,
        64 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 256 * 1 * 1,
        128 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 10,
        10,
    ],
    "tolstoi_char_rnn": [
        83 * 128,
        4 * (128 * 128 + 128 * 128),
        4 * 128,
        4 * (128 * 128 + 128 * 128),
        4 * 128,
        83 * 128,
        83,
    ],
    "two_d_beale": [1, 1],
    "two_d_branin": [1, 1],
    "two_d_rosenbrock": [1, 1],
}
