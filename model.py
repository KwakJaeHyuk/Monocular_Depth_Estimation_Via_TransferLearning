import sys

from tensorflow.keras.applications import densenet # Encoder에서 DenseNet의 사용을 위해
from keras.models import Model, load_model # Input과 output만으로 간단하게 모델을 생성할 수 있게 해준다, load_model :  모델을 불러올 때 사용
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate # 실질적인 layer 생성할 때 사용
from layers import BilinearUpSampling2D # upsampling 하는 레이어 설계
from loss import depth_loss_function # depth_loss_function class도 추후에 생성할 예정

def create_model(existing = '', is_twohundred=False, is_halffeatures=True):
    
    # 인코딩 부분
    if len(existing) == 0: # 만약 existing의 길이가 '0' 이라면 
        print('Loading base model (DenseNet)..') # print

        # Encoder Layers
        if is_twohundred: # 만약 200개이면 DenseNet201 사용(뭔소리지...)
            base_model = densenet.DenseNet201(input_shape = (None, None, 3), include_top = False)
            # base_model = applications.densenet.DenseNet201(input_shape = (None, None, 3), include_top = False)
        else:
            base_model = densenet.DenseNet169(input_shape = (None, None, 3), include_top = False) 
            # 최상단의 layer와의 연결은 안하겠다는 의미, include_top가 true일 때만 input_shape를 입력 안해도 된다.
            # input_shape는 3 channel를 사용하고 height와 width는 32 이하로 내려가면 안된다.     
            
        print('Base model loaded.')
        
        # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape # 최종적으로 나온 layer의 shape, decoder가 시작되는 Point

        for layer in base_model.layers: # 모든 layer를 train 가능한 상태로 전환
            layer.trainable = True
        
        # starting number of decoder filters 
        # decoding filter의 개수를 정해줌
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1]/2))
        else:
            decode_filters = int(base_model_output_shape[-1])

        # 실질적인 decoding 한 layer
        def upproject(tensor, filters, name, concat_with):
            
            up_i = BilinearUpSampling2D((2, 2), name = name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output])
            up_i = Conv2D(filters = filters, kernel_size=3, strides = 1, padding='same', name = name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            
            return up_i
        
        # decoding하는 모델의 설계
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)
        
        # UpSampling하는 과정에서 2배씩 filter 크기가 커져야 한다.
        # concat_with에서 pooling layer들은 densenet-169에 맞춰서 넣은 이름이다.
        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False: 
            decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')
            
        # Final exatracting layer
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)
        
        # Create the model
        # keras에서 제공하는 모델생성 함수
        model = Model(inputs=base_model.input, outputs=conv3)
    
    else:
        # 저장된 모델이 파일로 존재하는 경우
        if not existing.endswith('.h5'):
            sys.exit('저장된 모델이 없습니다.')
        custom_object = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function' : depth_loss_function}
        model = load_model(existing, custom_objects=custom_object)
        print('\nmodel loaded complete\n')
    
    print('\n모델 생성\n')
    
    return model
    
        
    
    
    



    
