# ----------- STATIC Variables -----------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CIFAR100_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR100_STD  = [0.2023, 0.1994, 0.2010] 

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

MNIST_MEAN = [0.1307]
MNIST_STD  = [0.3081] 


import torchvision.transforms as T
def get_default_transform(resize, crop, mean, std):
    transform = T.Compose([
                T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(crop),
                T.ToTensor(),
                T.Normalize(mean, std)
                ])
    return transform

# ----------- STATIC functions -----------------
import torchvision 
from .robust_wrapper import JigsawWrapper, BlurWrapper

def get_datasets(name, data_path, transform, flags=None, data_wrapper=None):
    # ---- Define the wrapper if required -----
    if data_wrapper is not None:
        robust_flag = flags.robust
        wrapper  = {
            'jigsaw': JigsawWrapper,
            "blur": BlurWrapper, 
        }[data_wrapper]
    # ------ CIFAR ---------
    if name =="cifar10":
        train_dataset  = torchvision.datasets.CIFAR10(root=data_path,  
                                                    train=True,  
                                                    download=True,
                                                    transform=transform) 
        valid_dataset  = torchvision.datasets.CIFAR10(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 

    elif name =="cifar100":
        train_dataset  = torchvision.datasets.CIFAR100(root=data_path,  
                                                    train=True,  
                                                    download=True,
                                                    transform=transform) 
        valid_dataset  = torchvision.datasets.CIFAR100(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 
            
    # ------ ImageNet ---------
    elif name =="imagenet1k":
        # train_dataset = torchvision.datasets.ImageNet(root=data_path, split="train", transform=transform)
        train_dataset = None  # need to be prepared in the future
        valid_dataset = torchvision.datasets.ImageNet(root=data_path, split="val", transform=transform)
    
    elif name == "mnist":
        from .mnist_perturb import PerturbedMNIST
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
        valid_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform, download=True) 

        train_dataset = PerturbedMNIST(train_dataset, flags.perturbed_mnist_epsilon)
        valid_dataset = PerturbedMNIST(valid_dataset, flags.perturbed_mnist_epsilon)

    elif name == "fashion_mnist":
        from .mnist_perturb import PerturbedMNIST
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=transform, download=True)
        valid_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=transform, download=True) 

        train_dataset = PerturbedMNIST(train_dataset, flags.perturbed_mnist_epsilon)
        valid_dataset = PerturbedMNIST(valid_dataset, flags.perturbed_mnist_epsilon)
    
    
    else:
        raise ValueError(f"{name} is not implemented data")
    
    if data_wrapper is not None:
        train_dataset = wrapper(train_dataset, transform, robust_flag)
        valid_dataset = wrapper(valid_dataset, transform, robust_flag)
    
    return train_dataset, valid_dataset