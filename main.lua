require 'torch';
require 'cutorch';
require 'paths';
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(0)
cutorch.manualSeedAll(0)

local opts = require 'opts'
local opt = opts.parse(arg)

local logger = optim.Logger(opt.logDir .. '/resnet.log')
logger:setNames{'train_loss', 'train_mse', 'val_loss', 'val_mse'}

local models = require 'models/init'
local DataLoader = require 'dataloader'
local Trainer = require 'train'

local valLoader, trainLoader = DataLoader.create(opt, {'train'})
local model, criterion = models.setup(opt)

print('trainLoader ', trainLoader)
print('valLoader ', valLoader)

local trainer = Trainer(model, criterion, opt, optimState)

for epoch = 1, opt.nEpochs do
    local train_loss, train_mse = trainer:train(epoch, trainLoader)
    local val_loss, val_mse = trainer:validate(epoch, valLoader)

    print("Scores are ", train_loss, train_mse, val_loss, val_mse);
    logger:add{train_loss, train_mse, val_loss, val_mse}

    -- code to sum/do operations with previous valStastistics and vs
    collectgarbage('count')
end

print("out of comment")

