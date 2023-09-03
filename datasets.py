import os
import gzip
import numpy as np

from urllib import request


class Dataset:
    """
    A dataset for recurrent neural networks.

    Arguments
    ---------
     - inputs: np.ndarray
        Input sequences of shape (seq_length, num_samples, input_size)
     - targets: np.ndarray
        Target sequences of shape (output_length, num_samples, output_size)
     - masks: np.ndarray
        Masks for the prediction steps of shape (seq_length, num_samples, 1)
     - regression: bool
        Problem type (True for regression, False for classification)
     - batch_first: bool
        Time axis (True for 0, False for 1)
    """

    def __init__(self, inputs, targets, masks, regression, batch_first=False):
        """
        See class documentation.
        """
        self.inputs = inputs
        self.targets = targets
        self.masks = masks
        self.regression = regression
        self.batch_first = batch_first

    @property
    def seq_length(self):
        return self.inputs.shape[1 if self.batch_first else 0]

    @property
    def output_length(self):
        return self.targets.shape[1 if self.batch_first else 0]

    @property
    def size(self):
        return self.inputs.shape[0 if self.batch_first else 1]

    @property
    def input_size(self):
        return self.inputs.shape[2]

    @property
    def output_size(self):
        return self.masks.shape[2]

    def train_test_split(self, test_ratio=None, test_size=None):
        """
        Split the dataset in train and test datasets according to `test_ratio`.
        """
        if test_ratio:
            split = int(self.size * (1.0 - test_ratio))

        elif test_size:
            split = self.size - test_size

        else:
            raise ValueError("Either `test_ratio` or `test_size` must be set.")

        if self.batch_first:
            train = Dataset(self.inputs[:split, :, :],
                            self.targets[:split, :, :],
                            self.masks[:split, :, :],
                            self.regression,
                            batch_first=True)
            test = Dataset(self.inputs[split:, :, :],
                           self.targets[split:, :, :],
                           self.masks[split:, :, :],
                           self.regression,
                           batch_first=True)

        else:
            train = Dataset(self.inputs[:, :split, :],
                            self.targets[:, :split, :],
                            self.masks[:, :split, :],
                            self.regression,
                            batch_first=False)
            test = Dataset(self.inputs[:, split:, :],
                           self.targets[:, split:, :],
                           self.masks[:, split:, :],
                           self.regression,
                           batch_first=False)

        return train, test


def copy_first_input(num_samples, seq_length, input_size=1, batch_first=False):
    """
    Generate the copy first input benchmark, as described in the paper.
    """
    inputs = np.random.randn(seq_length, num_samples, input_size)
    inputs = inputs.astype(np.float32)

    masks = np.zeros((seq_length, num_samples, 1), dtype=bool)
    masks[-1:, :, :] = True

    targets = inputs[:1, :, :]

    if batch_first:
        inputs = np.transpose(inputs, axes=(1, 0, 2))
        targets = np.transpose(targets, axes=(1, 0, 2))
        masks = np.transpose(masks, axes=(1, 0, 2))

    return Dataset(inputs, targets, masks, regression=True,
                   batch_first=batch_first)


def denoising(num_samples, seq_length, pad_length, input_size=1, number=5,
              batch_first=False):
    """
    Generate the denoising benchmark, as described in the paper.
    """
    inputs = np.empty((seq_length, num_samples, input_size + 1),
                      dtype=np.float32)
    inputs[:, :, :-1] = np.random.randn(seq_length, num_samples, input_size)
    inputs[:, :, -1] = np.zeros((seq_length, num_samples))
    inputs[- number, :, -1] = - 1.0

    masks = np.zeros((seq_length, num_samples, 1), dtype=bool)
    masks[- number:, :, :] = True

    targets = np.empty((number, num_samples, input_size), dtype=np.float32)

    num_selectable = seq_length - pad_length
    for i in range(num_samples):
        choices = np.random.choice(num_selectable, size=number, replace=False)
        inputs[choices, i, -1] = True
        targets[:, i, :] = inputs[np.sort(choices), i, :-1]

    if batch_first:
        inputs = np.transpose(inputs, axes=(1, 0, 2))
        targets = np.transpose(targets, axes=(1, 0, 2))
        masks = np.transpose(masks, axes=(1, 0, 2))

    return Dataset(inputs, targets, masks, regression=True,
                   batch_first=batch_first)


def sequential_mnist(train=True, black_pixels=0, batch_first=False):
    """
    Generate the sequential mnist benchmark, as described in the paper.
    Thanks to Hyeonseok Jung for his parsing of the MNSIT dataset.
        <https://github.com/hsjeong5/MNIST-for-Numpy>
    """
    source = "http://yann.lecun.com/exdb/mnist"
    destination = "mnist"

    if train:
        images = "train-images-idx3-ubyte.gz"
        labels = "train-labels-idx1-ubyte.gz"
    else:
        images = "t10k-images-idx3-ubyte.gz"
        labels = "t10k-labels-idx1-ubyte.gz"

    os.makedirs('mnist', exist_ok=True)

    images_filepath = f"{destination}/{images}"
    labels_filepath = f"{destination}/{labels}"

    # If archives are not saved, download them
    if not os.path.exists(images_filepath):
        try:
            request.urlretrieve(f"{source}/{images}", images_filepath)
        except Exception as e:
            print(f"{source}/{images} not found")
            raise e
    if not os.path.exists(labels_filepath):
        try:
            request.urlretrieve(f"{source}/{labels}", labels_filepath)
        except Exception as e:
            print(f"{source}/{labels} not found")
            raise e

    # Unzip images
    with gzip.open(images_filepath, "rb") as f:
        inputs = np.frombuffer(f.read(), np.uint8, offset=16)
        inputs = inputs.reshape(-1, 28 * 28, 1).transpose((1, 0, 2))
        inputs = (inputs / 255.0 - 0.5).astype(np.float32)

        num_samples = inputs.shape[1]

        if black_pixels > 0:
            padding = np.zeros((black_pixels, num_samples, 1),
                               dtype=np.float32)
            inputs = np.concatenate((inputs, padding), axis=0)

    # Unzip labels
    with gzip.open(labels_filepath, "rb") as f:
        targets = np.frombuffer(f.read(), np.uint8, offset=8)
        targets = targets.reshape(1, -1, 1).astype(np.long)

    # Create masks
    masks = np.zeros((28 * 28 + black_pixels, num_samples, 10), dtype=bool)
    masks[-1, :, :] = True

    shuffle = np.random.permutation(num_samples)
    inputs = inputs[:, shuffle, :]
    targets = targets[:, shuffle, :]
    masks = masks[:, shuffle, :]

    if batch_first:
        inputs = np.transpose(inputs, axes=(1, 0, 2))
        targets = np.transpose(targets, axes=(1, 0, 2))
        masks = np.transpose(masks, axes=(1, 0, 2))

    return Dataset(inputs, targets, masks, regression=False,
                   batch_first=batch_first)


PERMUTATION = np.array([
    598, 590, 209, 637, 174, 213, 429, 259, 593, 204, 576, 244, 235,
    218, 770, 155, 516,  67, 579, 109,  66, 522,  78, 473,  23, 211,
    706, 445, 644,  39, 332,  86, 137, 653, 656, 442, 525, 515, 334,
    630, 342, 780, 118, 652, 260, 779, 352, 432,  77, 691, 483, 682,
     49, 518, 168, 326, 377, 375, 568, 309, 629,  30, 361,  33,  31,
    627, 558, 405, 254, 412, 739, 486, 266, 331, 422, 231, 333, 357,
    620, 265,  54, 735, 514,  97, 506, 294, 234, 749, 311, 351, 120,
    436,  84,  10, 624, 464, 192, 530, 199,  29, 470, 323,  65, 350,
    659, 239,  81, 485, 291, 487, 264, 715, 535, 519,  76, 388, 523,
    570,  72, 693, 409, 208, 585,  63, 314, 672, 302, 750, 363, 393,
    752, 367, 705,   7, 533, 101, 428, 765, 745,   2, 398, 527, 764,
    196, 641, 729, 493, 448, 215, 425, 581, 417,  79, 148, 335, 247,
    559, 762, 133, 648,  55, 411, 597, 675, 545, 617, 720, 296, 362,
     60, 741, 360, 440, 662, 426, 383, 327, 584, 286,  90, 382, 181,
    443, 618, 158,  69, 446, 131,  44,  70, 210, 340, 300, 275, 135,
    740, 165, 164,  28, 639, 193, 220, 534, 306, 136, 521, 299, 140,
    457,   6, 611, 478,  73, 250, 778, 145, 281, 290, 434, 132, 771,
    539, 734, 615,  41, 477, 108, 628,  56, 292, 704, 394, 227, 212,
    583, 319,  24, 467, 733, 336, 365, 544, 110,  82,  51, 465, 731,
    718, 632, 198, 549, 687, 499, 482, 479, 139, 444, 420,  18, 649,
    328,  83,  61, 572, 431, 182, 481, 223, 433, 451, 381, 453, 721,
    746, 176, 536, 626, 163, 248, 507, 696,  74, 616, 713, 104, 114,
    424,  92, 395,  89, 751, 495, 728, 609, 594,  11, 338,  43,  42,
    167, 689, 603, 396, 178, 688, 529, 177, 543, 726, 257, 344, 456,
     15, 606, 256, 355, 517, 324, 462, 708, 356, 329, 605,   9, 249,
     22, 221, 537, 676, 768, 439, 657, 203, 237,  93, 680, 346, 490,
    284, 184, 636, 380, 153,  75, 512, 277,  68, 494, 188, 271, 236,
     88, 667, 117, 125, 736, 289, 238,   0, 775, 368, 743, 450, 278,
    776, 116, 228, 634, 404, 677, 274, 318, 541, 144, 497, 678, 711,
    575, 369, 268, 557, 307, 310, 782,  46, 349, 371, 513, 261, 195,
    783, 658, 107,  59, 589, 423, 100, 660, 703, 633, 586, 179, 304,
    761, 650, 755, 149, 124, 623, 683, 185, 531,  50, 500, 773, 722,
    321, 353, 724, 142, 370, 141, 399, 511, 320,  19, 172, 640, 312,
    390, 730,  12, 407, 408, 305, 354,  25, 587, 169,  38, 175, 245,
    298, 654, 416, 538, 272, 601, 154, 126, 449, 716, 341, 430, 287,
    113, 501, 173, 359, 774,  57, 542, 222, 280,  17, 127, 322, 255,
    528, 588, 468, 753, 190, 115, 695, 645,  94, 180, 301, 571, 580,
    551, 548, 694, 532,   5, 769,  45, 710, 157, 595, 171,  16,  48,
    759, 719,   3, 567, 554, 316, 552, 480, 447, 723, 283,  96, 285,
    526, 225,  26, 631, 263, 437, 364, 229,  37, 754, 374, 469, 756,
    668, 582, 194, 670, 679, 503, 758, 655, 757, 162, 604, 152, 547,
    742, 602, 111, 226, 651, 103, 421, 419, 119,  53, 151, 403, 738,
    207, 767, 608,   8, 638,  36, 452, 253, 303, 596, 569, 635, 262,
    297, 414, 150, 625, 698, 550, 488, 147, 146, 578, 727, 591, 348,
    463, 325, 186, 123, 669, 143, 748, 197, 279, 293, 400, 122, 183,
    202, 438, 246, 415, 697, 129, 402, 621, 613, 712, 219, 714, 599,
    717, 610, 386, 760, 509, 267, 685, 441, 496, 112, 232, 684, 607,
    373, 233, 622, 317, 410, 709, 358, 258, 282, 376, 384, 224, 744,
    643, 472, 347, 505, 772, 725, 707, 619, 671, 664, 556, 577,  85,
    242, 159, 524,  35, 540, 170, 673, 665, 737,  95, 563, 240, 574,
    460, 553, 690, 206, 392, 397, 666, 217,   4, 642, 701, 612, 546,
     98, 573, 406, 502,  47,  32, 200, 134,  27, 692, 230, 489, 378,
    288, 418, 674, 391, 592, 498, 138,  62, 471, 647, 128, 763, 520,
     64,  14, 156,  40, 492, 379, 187, 216,  52, 337, 295, 251, 461,
    455, 781, 269, 201, 161, 555, 401, 702, 476, 105, 565, 389,   1,
    732, 561,  80, 205,  34, 508, 427, 454, 366,  91, 339, 564, 345,
    241,  13, 315, 600, 387, 273, 166, 777, 646, 484, 766, 504, 243,
    566, 562, 686, 189, 699, 475, 681, 510,  58, 474, 560, 747, 252,
     21, 313, 459, 160, 276, 191, 385, 413, 491, 343, 308, 661, 130,
    663,  99, 372,  87, 458, 330, 214, 466, 121, 614,  20, 700,  71,
    106, 270, 435, 102
])


def permuted_mnist(train=True, black_pixels=0, batch_first=False):
    """
    Generate the permuted mnist benchmark, as described in the paper.
    """
    dataset = sequential_mnist(train=train, black_pixels=0)

    inputs = dataset.inputs[PERMUTATION, :, :]
    num_samples = inputs.shape[1]

    if black_pixels > 0:
        padding = np.zeros((black_pixels, num_samples, 1),
                           dtype=np.float32)
        inputs = np.concatenate((inputs, padding), axis=0)

    masks = np.zeros((28 * 28 + black_pixels, num_samples, 10), dtype=bool)
    masks[-1, :, :] = True

    if batch_first:
        inputs = np.transpose(inputs, axes=(1, 0, 2))
        dataset.targets = np.transpose(dataset.targets, axes=(1, 0, 2))
        masks = np.transpose(masks, axes=(1, 0, 2))

    return Dataset(inputs, dataset.targets, masks,
                   regression=False, batch_first=batch_first)


def row_mnist(train=True, black_pixels=0, batch_first=False):
    """
    Generate the row mnist benchmark, as described in the paper.
    """
    dataset = permuted_mnist(train=train, black_pixels=0)

    inputs = dataset.inputs
    num_samples = inputs.shape[1]
    inputs = inputs.transpose((1, 0, 2)).reshape((-1, 28, 28)).transpose((1, 0, 2))

    if black_pixels > 0:
        padding = np.zeros((black_pixels, num_samples, 28),
                           dtype=np.float32)
        inputs = np.concatenate((inputs, padding), axis=0)

    masks = np.zeros((28 + black_pixels, num_samples, 10), dtype=bool)
    masks[-1, :, :] = True

    if batch_first:
        inputs = np.transpose(inputs, axes=(1, 0, 2))
        dataset.targets = np.transpose(dataset.targets, axes=(1, 0, 2))
        masks = np.transpose(masks, axes=(1, 0, 2))

    return Dataset(inputs, dataset.targets, masks, regression=False,
                   batch_first=batch_first)
