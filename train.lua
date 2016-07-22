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
    self.confusion = optim.ConfusionMatrix({1,2,3,4,5})
end

function Trainer:train(epoch, dataLoader)
    self.optimState.learningRate = self:learningRate(epoch)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local N = 0
    local kappa_sum, loss_sum = 0.0, 0.0

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
        local loss = self.criterion:forward(predictions, (self.target + 1))

        self.model:zeroGradParameters()
        self.criterion:backward(predictions, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        optim.sgd(feval, self.params, self.optimState)

        loss_sum = loss_sum + loss
        kappa_sum = kappa_sum + self:computeKappa(predictions, (self.target + 1))

        N = N + 1

        assert(self.params:storage() == self.model:parameters()[1]:storage(), 'Storage changed')

        timer:reset()
        dataTimer:reset()
        xlua.progress(n, trainSize)
    end
    return loss_sum/N, kappa_sum/N
end

function Trainer:validate(epoch, dataLoader)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local valSize = dataLoader:size()

    local N = 0
    local kappa_sum, loss_sum = 0.0, 0.0

    self.model:evaluate()
    for n, sample in dataLoader:run(epoch) do
        local dataTime = dataTimer:time().real
        self:copyInputs(sample)

        local output = self.model:forward(self.input):float()
        local _, predictions = output:float():sort(2, true)
        predictions = predictions:narrow(2,1,1):double():cuda()

        local loss = self.criterion:forward(predictions, self.target)
        
        loss_sum = loss_sum + loss
        kappa_sum = kappa_sum + self:computeKappa(predictions, (self.target + 1))

        N = N+1
        xlua.progress(n, valSize)
    end
    return loss_sum/N, kappa_sum/N
end

function Trainer:computeKappa(predictions, target)
    local confusion = self.confusion
    confusion:zero()

    -- print('predictions ', predictions)
    -- print('target ', target)
    confusion:batchAdd(predictions, target)
    --confusion:batchAdd(target, target)
    local mat = confusion.mat:double()
    local N = mat:size(1)
    local tmp = torch.range(1, N):view(1, N)
    local tmp1 = torch.range(1, N):view(N, 1)
    local W= tmp:expandAs(mat)-tmp1:expandAs(mat)
    W:cmul(W)
    W:div((N-1)*(N-1))
    local total = mat:sum()
    local row_sum = mat:sum(1)/total
    local col_sum = mat:sum(2)
    local E = torch.cmul(row_sum:expandAs(mat), col_sum:expandAs(mat)):double()
    local kappa = 1 - torch.cmul(W, mat):sum()/ torch.cmul(W, E):sum()
    return kappa
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
