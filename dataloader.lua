local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}

local DataLoader = torch.class('eyeart.DataLoader', M)
local KaggleDR = require 'datasets/kaggledr'
local script = paths.dofile('datasets/kaggledr-gen.lua')

function DataLoader.create(opt, setNames)
    local loaders = {}

    for i, setName in ipairs(setNames) do
        local dataSets = script.exec(opt, setName)
        for i, dataSet in ipairs(dataSets) do
            loaders[i] = M.DataLoader(dataSet, opt)
        end
    end

    return table.unpack(loaders)
end

function M.DataLoader:__init(data, opt)

    local manualSeed = opt.manualSeed
    local function init()
        require('datasets/kaggledr')
    end
    local function main(idx)
        if manualSeed ~= 0 then
            torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        dataSet = data
        return dataSet:size()
    end

    local pool, sizes = Threads(opt.nThreads, init, main)
    self.setName = data:getSplitName()
    self.pool = pool
    self.dataSet = data
    self.__size = sizes[1][1]
    self.batchSize = opt.batchSize
end

function M.DataLoader:run(epoch)
    local pool = self.pool
    local size, batchSize = self.__size, self.batchSize
    print('batchSize is ', batchSize, ' self.__size is ', size)
    -- local perm = torch.randperm(size)
    local perm = self.dataSet:get_image_indicies(epoch)

    local idx, sample = 1, nil

    local function enqueue()
        while idx <= size and pool:acceptsjob() do
            local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
            -- print('max of indices is ', indices:max(), 'min is ', indices:min(), indices:size(1), self.dataSet:size())
            pool:addjob(
                function(indices)
                    local sz = indices:size(1)
                    local batch, imageSize
                    local target = torch.IntTensor(sz)
                    for i, idx in ipairs(indices:totable()) do
                        local sample = dataSet:get(idx)
                        local input = dataSet:preprocess(sample.input)
                        if not batch then
                            imageSize = input:size():totable()
                            batch = torch.FloatTensor(sz, table.unpack(imageSize))
                        end
                        batch[i]:copy(input)
                        target[i] = sample.target
                    end
                    collectgarbage()
                    return {
                        input = batch,
                        target = target
                    }
                end,
                function(result)
                    sample = result
                end,
                indices
            )
            idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not pool:hasjob() then
            return nil
        end
        pool:dojob()
        if  pool:haserror() then
            pool:synchronize()
        end
        enqueue()
        n = n+1
        return n, sample
    end

    return loop
end

function M.DataLoader:size()
    return math.ceil(self.__size/self.batchSize)
end

function M.DataLoader:__tostring()
    return 'DataLoader@' .. self.setName:gsub("^%l", string.upper)
end

return M.DataLoader
