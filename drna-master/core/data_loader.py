import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import sys
sys.path.append('..')
from config import INPUT_SIZE

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.transform = transform
        self.data_root = data_root
       # print(self.data_root)
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.labels = []

        for data in data_list:
            image_path = data[:-1]
            label = image_path.split('/')[1]
            self.img_paths.append(image_path)
            self.labels.append(label)

    def __getitem__(self, item):

     #   print(self.data_root)
        img_path, label= self.img_paths[item], self.labels[item]
        #img_path_full = os.path.join(self.data_root, img_path)
        img_path_full = self.data_root+img_path
      #  print(img_path_full)
        img = Image.open(img_path_full).convert('RGB')
        # label = np.array(label,dtype='float32')
        label = int(label)-100
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.n_data

if __name__ == "__main__":
    train_transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
    train_dataset = GetLoader(data_root='../../data/image_resize_448/', 
                         data_list='../train.txt',
                         transform=train_transform)
    print(len(train_dataset))
    print(len(train_dataset.labels))
    for data in train_dataset:
        print(data[0].size(), data[1])
    test_dataset = GetLoader(data_root='../../data/image_resize_448/', 
                    data_list='../test.txt',
                    transform=test_transform)
    print(len(test_dataset))
    print(len(test_dataset.labels))
    for data in test_dataset:
        print(data[0].size(), data[1])
