require 'torch';
require 'cutorch';
require 'paths';

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(0)
cutorch.manualSeedAll(0)

local opts = require 'opts'
local opt = opts.parse(arg)

local models = require 'models/init'
local DataLoader = require 'dataloader'
local Trainer = require 'train'

local trainLoader = DataLoader.create(opt, {'train'})
local model, criterion = models.setup(opt)

print(trainLoader)

local trainer = Trainer(model, criterion, opt, optimState)

for epoch = 1, opt.nEpochs do
    local trainStatistics = trainer:train(epoch, trainLoader)
    local vs = trainer:validate(epoch, trainLoader)

    -- code to sum/do operations with previous valStastistics and vs
    collectgarbage('count')
end

print("out of comment")

