local optim = require 'optim'

local M = {}
local Trainer = torch.class('eyeart.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay
    }
    self.opt = opt
    self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataLoader)
    self.optimState.learningRate = self:learningRate(epoch)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local N = 0
    local mean_sq_sum, loss_sum = 0.0, 0.0

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local trainSize = dataLoader:size()

    print('=> Training epoch #'.. epoch)

    self.model:training()
    for n, sample in dataLoader:run(epoch) do

        local dataTime = dataTimer:time().real

        self:copyInputs(sample)

        local output = self.model:forward(self.input):float()
        local _, predictions = output:float():sort(2, true)
        predictions = predictions:narrow(2,1,1):double():cuda()
        -- print(predictions)
        -- print(self.target)
        local loss = self.criterion:forward(predictions, (self.target + 1))

        -- print(output)

        --[[--[
        for i = 0, 5 do 
            print(i, output:eq(i):sum(), self.target:eq(i):sum())
        end
        --]]--]

        self.model:zeroGradParameters()
        self.criterion:backward(predictions, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        optim.sgd(feval, self.params, self.optimState)

        -- local mean_sq_err = self:computeScore(output, (sample.target + 1))

        -- mean_sq_sum = mean_sq_sum + mean_sq_err
        loss_sum = loss_sum + loss

        N = N + 1

        assert(self.params:storage() == self.model:parameters()[1]:storage(), 'Storage changed')

        timer:reset()
        dataTimer:reset()
        xlua.progress(n, trainSize)
    end
    return loss_sum/N -- , mean_sq_sum/N
end

function Trainer:validate(epoch, dataLoader)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local valSize = dataLoader:size()

    local N = 0
    local mean_sq_sum, loss_sum = 0.0, 0.0

    self.model:evaluate()
    for n, sample in dataLoader:run(epoch) do
        local dataTime = dataTimer:time().real
        self:copyInputs(sample)

        local output = self.model:forward(self.input):float()
        local _, predictions = output:float():sort(2, true)
        predictions = predictions:narrow(2,1,1):double():cuda()

        local loss = self.criterion:forward(predictions, self.target)
        
        -- local mean_sq_err = self:computeScore(output, sample.target)

        -- mean_sq_sum = mean_sq_sum + mean_sq_err
        loss_sum = loss_sum + loss

        N = N+1
        xlua.progress(n, valSize)
    end
    return loss_sum/N -- , mean_sq_sum/N
end

function Trainer:computeScore(output, target)
    -- print(output)
    local batchSize = output:size(1)
    local _, predictions = output:float():sort(2, true)

    -- local correct = predictions:eq(target:add(1):long():view(batchSize, 1):expandAs(output))

    local mean_square  = torch.pow((predictions:narrow(2,1,1):double() - target:view(batchSize, 1):double()), 2):sum()

    return mean_square / batchSize
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
