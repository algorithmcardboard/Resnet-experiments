require 'torch';
require 'paths';
require 'cutorch';
require 'cunn';
require 'cudnn';
require 'optim';

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(0)
cutorch.manualSeedAll(0)

local opts = require 'opts'
local opt = opts.parse(arg)

local models = require 'models/init'
local DataLoader = require 'dataloader'

local trainLoader = DataLoader.create(opt, {'train'})
local model, criterion = models.setup(opt)

print(trainLoader)

--[===[
for epoch = startEpoch, opt.nEpochs do
    local kFold = trainLoader.getKFold()
    local valStastistics;
    for k = 1, kFold do
        local trainStatistics = trainer:train(epoch, trainLoader:getTrainSet(k))
        local vs = trainer:val(epoch, trainLoader:getValidationSet(k))

        -- code to sum/do operations with previous valStastistics and vs
    end
    collectgarbage('count')
end
--]===]

print("out of comment")

