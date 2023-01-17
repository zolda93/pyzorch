from zorch import np,Tensor
import random

try:
    import Image
except ImportError:
    from PIL import Image



class Compose:
    def __init__(self,transforms=[]):
        assert isinstance(transforms,list),"[*] transforms needs to be list of transforms"

        self.transforms = transforms

    def __call__(self,img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)

        return img


class Resize:
    def __init__(self,size,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self,img):
        w,h = img.size
        return img.resize((self.size,int(self.size*h/w)),self.interpolation)


class CenterCrop:
    def __init__(self,size):
        self.size = size

    def __call__(self,img):
        w,h = img.size
        left = (w-self.size)//2
        right = w-((w-self.size)//2+(w-self.size)%2)
        up = (h-self.size)//2
        bottom = h-((h-self.size)//2+(h-self.size)%2)
        return img.crop((left, up, right, bottom))


class ToTensor:
    def __init__(self):
        pass

    def __call__(self,img):

        if isinstance(img,Image.Image):
            img = np.asarray(img)
            img = img.transpose(2,0,1)
            img = img.reshape(1,*img.shape) / 255.
        elif isinstance(img,np.ndarray):
            img = img/255.
        else:
            raise ValueError

        return Tensor(img)


class Normalize:
    def __init__(self,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,img):
        if img.shape[1] == 3:
            img.data[:,0] = (img.data[:,0]-self.mean[0])/self.std[0]
            img.data[:,1] = (img.data[:,1]-self.mean[1])/self.std[1]
            img.data[:,2] = (img.data[:,2]-self.mean[2])/self.std[2]
        else:
            img.data[:,0] = (img.data[:,0]-self.mean[0])/self.std[0]
        return img


class RandomHorizontalFlip:

    def __init__(self,p=0.5):
        self.p = p


    def __call__(self,img):

        if random.random() < self.p:
            if isinstance(img,np.ndarray):
                return img[:,:,:,::-1]
            elif isinstance(img,Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip:

    def __init__(self,p=0.5):
        self.p = p


    def __call__(self,img):

        if random.random() < self.p:
            if isinstance(img,np.ndarray):
                return img[:,:,::-1,:]
            elif isinstance(img,Image.Image):
                return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


            
