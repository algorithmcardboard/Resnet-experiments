local image = require 'image'
require 'optim';


local IMAGE_DIR = '/scratch/data/cifar-10/images/'
local CIFAR_FILE = '/scratch/data/cifar-10/cifar10.t7'
local cifar = torch.load(CIFAR_FILE)

local training_set, data, labels = cifar.train, cifar.train.data, cifar.train.labels

local fileName

local outfile = torch.LongTensor(data:size(1), 2)

print('data size is ', data:size(1))
for i = 1, data:size(1) do 
    fileName = IMAGE_DIR .. i .. '.jpg'
    image.save(fileName, data[i])
    outfile[i][1] = i
    outfile[i][2] = labels[i]
    print('iteration ', i, labels[i])
end

torch.save('/scratch/data/cifar-10/trainLabels.t7', outfile)
