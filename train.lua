local optim = require 'optim'

local M = {}
local Trainer = torch.class('eyeart.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or {
        learningRate = opt.LR,
        learningRateDecay = 0.01,
        momentum = opt.momentum,
        nesterov = true,
        dampning = 0.0,
        weightDecay = opt.weightDecay
    }
    self.opt = opt
    self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataLoader)
    self.optimState.learningRate = self:learningRate(epoch)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local trainSize = dataLoader:size()

    print('=> Training epoch #'.. epoch)

    self.model:training()
    for n, sample in dataLoader:run() do
        local dataTime = dataTimer:time().real
        self:copyInputs(sample)

        local output = self.model:forward(self.input):float()
        local loss = self.criterion:forward(self.model.output, self.target)

    end
end

function Trainer:validate(epoch, dataLoader)
    print('calling validate')
end

function Trainer:learningRate(epoch)
    local decay = math.floor((epoch - 1)/30)
    return self.opt.LR * math.pow(0.1, decay)
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or torch.CudaTensor()
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end


return M.Trainer
