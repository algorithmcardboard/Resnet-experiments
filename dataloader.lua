local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}

local DataLoader = torch.class('eyeart.DataLoader', M)
local KaggleDR = require 'datasets/kaggledr'

function DataLoader.create(opt, setNames)
    local loaders = {}

    for i, setName in ipairs(setNames) do
        local dataSet = KaggleDR.create(opt, setName)
        loaders[i] = M.DataLoader(dataset, opt, setName)
    end

    return table.unpack(loaders)
end

function M.DataLoader:__init(dataset, opt, setName)
    self.setName = setName
    self.dataset = dataset
end

function M.DataLoader:__tostring()
    return 'DataLoader@' .. self.setName:gsub("^%l", string.upper)
end

return M.DataLoader
