from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

# 난수 시작 시점 지정
random.seed(0)

# Data augment시켜주는 클라스 작성

class BasicPolicy(object):
    def __init__(self, mirror_ratio = 0, flip_ratio = 0, color_change_ratio = 0, is_full_set_colors = False, add_noise_peak = 0.0, erase_ratio = -1.0):
        
        
        from itertools import product, permutations
        # from itertools import permutations
        # _list = [3, 2, 1]
        # perm = list(permutations(_list, 3))
        # print(perm)
        # [(3, 2, 1), (3, 1, 2), (2, 3, 1), (2, 1, 3), (1, 3, 2), (1, 2, 3)]
        
        # from itertools import permutations, product
        # _list = ['!@', 'asd']
        # pd = list(product(*_list))
        # print(pd)
        # *_list로 표현하게 된다면 리스트안에 있는 문자열을 모두 조합하게 된다.
        # [('!', 'a'), ('!', 's'), ('!', 'd'), ('@', 'a'), ('@', 's'), ('@', 'd')]
        
        # 만약 is_full_set_colors가 True라면 product연산을 진행 => repeat=3 이기 때문에 최대 3개까지 중첩가능
        # 조건문에 만족하지 않는다면 permutation에서 (0,1,2), 3하고의 조합을 만들어 낸다.
        self.indices = list(product([0,1,2], repeat = 3)) if is_full_set_colors else list(permutations(range(3), 3))
        
        # insert(i, x) => 원하는 위치 i에 x를 삽입한다. 
        self.indices.insert(0, [0, 1, 2])
        self.add_noise_peak = add_noise_peak
        
        # Mirror and flip
        self.color_change_ratio = color_change_ratio
        self.mirror_ratio = mirror_ratio
        self.flip_ratio = flip_ratio
        
        # Erase
        self.erase_ratio = erase_ratio
        
    def __call__(self, img, depth):
        if self.add_noise_peak > 0:
            PEAK = self.add_noise_peak
            img = np.random.poisson(np.clip(img, 0, 1) * PEAK) / PEAK
        
        # 1) Color change
        policy_idx = random.randint(0, len(self.indices) - 1)
        if random.uniform(0, 1) >= self.color_change_ratio:
            policy_idx = 0

        img = img[...,list(self.indices[policy_idx])]

        # 2) Mirror image
        if random.uniform(0, 1) <= self.mirror_ratio:
            img = img[...,::-1,:]
            depth = depth[...,::-1,:]

        # 3) Flip image vertically
        if random.uniform(0, 1) < self.flip_ratio:
            img = img[...,::-1,:,:]
            depth = depth[...,::-1,:,:]

        # 4) Erase random box
        if random.uniform(0, 1) < self.erase_ratio:
            img = self.eraser(img)

        return img, depth

    def __repr__(self):
        return "Basic Policy"

    def eraser(self, input_img, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=True):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img
        
    def debug_img(self, img, depth, idx, i, prefix=''):
        from PIL import Image
        aug_img = Image.fromarray(np.clip(np.uint8(img*255), 0, 255))
        aug_img.save(prefix+str(idx)+"_"+str(i)+'.jpg',quality=99)
        aug_img = Image.fromarray(np.clip(np.uint8(np.tile(depth*255,3)), 0, 255))
        aug_img.save(prefix+str(idx)+"_"+str(i)+'.depth.jpg',quality=99)