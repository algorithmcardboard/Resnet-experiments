require 'torch';
require 'paths';
require 'cutorch';
require 'cunn';
require 'cudnn';
require 'optim';

local opts = require 'opts'


torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)


local opt = opts.parse(arg)

torch.manualSeed(0)
cutorch.manualSeedAll(0)
