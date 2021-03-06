from unicodedata import name
import numpy as np
from util import DepthNorm
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from tensorflow.keras.utils import Sequence
from augment import BasicPolicy
import csv 

# Zipfile class를 사용하여 zip파일 내부의 파일들을 read()를 통해 읽어온다.
def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    print(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def read_file(input_file):
    return {name: input_file.read(name) for name in input_file.namelist()}

# 이미지의 사이즈 조정하는 함수
# resize(이미지, spline보간을 위한 값, 원래 범위를 가지고 있을 것인지, 어떠한 방식으로 이미지를 바꿀 것인지, 가우시안 필터 사용유무)
def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True)

# zip파일을 불러옴
def get_nyu_data(batch_size, nyu_data_zipfile='nyu_data.zip'):
    data = extract_zip(nyu_data_zipfile)
    # data = open('nyu_data')
    
    nyu2_train = list()
    
    # decode :  utf-8 형식을 Byte형식으로 변경
    # encode :  Byte형식을 utf-8 형식으로 변경
    # csv형식으로 저장되어 있는 파일들을 ',' 단위로 나눠준다.
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split("\n") if len(row) > 0))
    
    # rgb이미지와 depth이미지의 batch_size와 tensor를 지정해준다.
    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)
    
    if False:
        nyu2_train = nyu2_train[:10]
        nyu2_test = nyu2_test[:10]
        
    return data, nyu2_train, nyu2_test, shape_rgb, shape_depth

def get_nyu_train_test_data(batch_size):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size)
    
    # 뒤에 두개의 클라스 생성 후 분석 ㄱㄱ
    train_generator = NYU_BasicAugmentRGBSequence(data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = NYU_BasicRGBSequence(data, nyu2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    
    return train_generator, test_generator

##################################################################

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]
            # print(sample)
            # img = Image.open("C:/Users/MSI/Desktop/2022_BARAM_First_Semester/nyu_data.zip/data/nyu2_train/bathroom_0053_out/66.png")
            # img.show()
            x = np.clip(np.asarray(Image.open( "C:/Users/MSI/Desktop/2022_BARAM_First_Semester/nyu_data/"+sample[0] )).reshape(480,640,3)/255,0,1)
            y = np.clip(np.asarray(Image.open( "C:/Users/MSI/Desktop/2022_BARAM_First_Semester/nyu_data/"+sample[1] )).reshape(480,640,1)/255*self.maxDepth,0,self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class NYU_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( "C:/Users/MSI/Desktop/2022_BARAM_First_Semester/nyu_data/"+sample[0])).reshape(480,640,3)/255,0,1)
            y = np.asarray(Image.open( "C:/Users/MSI/Desktop/2022_BARAM_First_Semester/nyu_data/"+sample[1]), dtype=np.float32).reshape(480,640,1).copy().astype(float) / 10.0
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y