local image = require 'image';

local M = {}

local KaggleDR = torch.class('eyeart.KaggleDR', M)


function KaggleDR:__init(opt, split, info)
    self.split = split
    self.dir = opt.data
    self.info = info
    self.__size = info.size
end

function KaggleDR:get(id)
    local data = self.info.data
    -- print('fetching ', id, 'data is ', data:size())

    local encounter_id, pos_id, dr_level = table.unpack(data[id]:totable())
    local fileName = self.dir .. encounter_id .. '_' .. (pos_id == 1 and 'left' or 'right') .. '.jpg'
    -- print(id, encounter_id, pos_id, dr_level, fileName)

    img = image.load(fileName, 3, 'float')
    return {
        input = img,
        target = dr_level
    }
end

function KaggleDR:size()
    return self.__size
end

function KaggleDR:preprocess(img)
    return img
end

function KaggleDR:getSplitName()
    return self.split
end


return M.KaggleDR
