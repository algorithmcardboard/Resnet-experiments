local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}

local DataLoader = torch.class('eyeart.DataLoader', M)
local KaggleDR = require 'datasets/kaggledr'

function DataLoader.create(opt, setNames)
    local loaders = {}

    for i, setName in ipairs(setNames) do
        local dataSet = KaggleDR.create(opt, setName)
        loaders[i] = M.DataLoader(dataSet, opt, setName)
    end

    return table.unpack(loaders)
end

function M.DataLoader:__init(dataSet, opt, setName)

    local manualSeed = opt.manualSeed
    local function init()
        require('datasets/kaggledr')
    end
    local function main(idx)
        if manualSeed ~= 0 then
            torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        return dataSet:size()
    end

    local pool, sizes = Threads(opt.nThreads, init, main)
    self.setName = setName
    self.dataSet = dataSet
    self.pool = pool
    self.__size = sizes[1][1]
    print("Assigning sizes ", sizes)
    self.batchSize = opt.batchSize
end

function M.DataLoader:run()

    local pool = self.pool

    local function enqueue()

    end

    local function loop()
        enqueue()
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
