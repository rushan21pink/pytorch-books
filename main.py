import time
startTime = time.time()
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from torch.autograd import Variable
import torchvision
import os


dirname = './img/'
files = os.listdir(dirname)


transforms = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
                                ])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model2 = torchvision.models.resnet50()
model2.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))
model2.load_state_dict(torch.load('../model_2.5.pth'))
model2.eval()
model2 = model2.to(device)
# None


def f(path, model, transforms, device):
    with torch.no_grad():
        img = Image.open(path)
        tensor = transforms(img).unsqueeze(0).to(device)
        logit = model(tensor).detach().cpu()
        return torch.softmax(logit, dim=1)[0][1]


result = f('/home/books/img/ayaz2.jpg', model2, transforms, device)
print(result)

endTime = time.time()
elapsedTime = endTime - startTime 
print("Elapsed Time = %s" % elapsedTime)


# print(files)

# for elem in files:
#   print('------------------------------')

#   result = f(f'{dirname}{elem}', model2, transforms, device)
  
#   print('книга ' + str(elem) + ' в хорошем состоянии на ' + str(result) + '%')