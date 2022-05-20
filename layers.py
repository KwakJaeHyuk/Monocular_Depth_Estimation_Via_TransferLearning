from keras.layers import Layer, InputSpec
import keras.utils.conv_utils as conv_utils
import tensorflow as tf
import keras.backend as K # keras에서는 저수준 연산들을 제공하지 않는다.(텐서곱, 합성곱 ...) 따라서 그 연산들이 가능하게 해주는
                          # 따라서 그 연산들이 가능하게 해주는 backend 라이브러리를 사용
                          
def normalize_data_format(value):
    if value is None:
        value = K.image_data_format() # 만약 value가 None이라면 image_data_format을 입력(image_data_format은 'channels_first'와 'channels_last'를 리턴)
        
    data_format = value.lower() # 문자열을 소문자로 바꾸는 파이썬 함수 - lower()
    if data_format not in {'channels_first', 'channels_last'}: # data_format에 올바른 문자열이 들어갔는지 확인을 위한 조건문
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last", Received: ' + 
                         str(value)) # 만약 'channels_first', 'channels_last'의 값이 안들어가 있다면 ValueError를 출력
    return data_format # 모든 예외처리를 거치고 최종 data_format을 리턴

class BilinearUpSampling2D(Layer):
    def __init__(self, size = (2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs) # 부모 class를 불러오는 함수 super().__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4) # 인풋 레이어 차원을 4차원으로 설정
        
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first': # 만약 인풋 데이터가 첫번째 채널일 경우
            # 처음보는 if문 형태 => 만약 if문을 만족하게 된다면 height변수 안에 size입력
            # 따라서 input_shape[2]가 None이 아니라면 height = self.size[0] * input_shape[2]을 만족하게 될 것이고 input_shape[2]가 None이라면 height는 None이 될 것이다.
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None 
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            
            return (input_shape[0],
                    input_shape[1],
                    height, 
                    width)
            
        elif self.data_format == 'channels_last': # 만약 인풋 데이터가 마지막 채널일 경우
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])
            
    # 보간해주는 과정인듯        
    def call(self, inputs): 
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        
        # 선형 보간법을 사용하여 이미지를 보간한다.
        return tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    
    # 딕셔너리로 size와 data_format을 분리한다.
    def get_config(self):
        
        config = {'size' : self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config() 
        
        # items() 함수를 사용하여 딕셔너리 값을 쌍으로 얻을 수 있다.
        return dict(list(base_config.items()) + list(config.items())) 
            