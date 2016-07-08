local image = require 'image';

local M = {}

local KaggleDR = torch.class('eyeart.KaggleDR', M)


function KaggleDR:__init(opt, split, info)
    self.split = split
    self.dir = opt.data
    self.info = info
    self.__size = info.size
end

function KaggleDR:size()
    return self.__size
end

function KaggleDR:getSplitName()
    return self.split
end


return M.KaggleDR
