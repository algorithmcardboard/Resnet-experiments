require 'torch';
require 'paths';
require 'cutorch';
require 'cunn';
require 'cudnn';
require 'optim';

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opts = require 'opts'
local models = require 'models/init'
local DataLoader = require 'dataloader'

local opt = opts.parse(arg)

torch.manualSeed(0)
cutorch.manualSeedAll(0)

local model, criterion = models.setup(opt)

local trainLoader, valLoader = DataLoader.create(opt)
