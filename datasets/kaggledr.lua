local image = require 'image';
local t = require 'datasets/transforms'

local M = {}

local KaggleDR = torch.class('eyeart.KaggleDR', M)


function KaggleDR:__init(opt, split, info)
    self.split = split
    self.dir = opt.data
    self.info = info
    if split == 'train' then
        self.classDistribution = info.classDistribution
    end
    self.__size = info.size
    self.R = 0.975

    self.initialWeights = torch.Tensor({1.36, 14.4, 6.64, 40.2, 49.6})
    self.finalWeights = torch.Tensor({1, 2, 2, 2, 2})
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

function KaggleDR:get_image_indicies(epoch)
    if self.split ~= 'train' then
        print('returning non train permutation "'.. self.split .. '"')
        return torch.randperm(self.__size)
    end

    -- do this only for training set

    local weights = self.R^(epoch -1) * self.initialWeights + (1 - self.R^(epoch - 1)) * self.finalWeights
    weights = torch.cmul(weights, self.classDistribution)
    print(weights)

    dr_levels = torch.multinomial(weights, self.__size, true)

    for i = 1, 5 do print(dr_levels:eq(i):sum()) end

    return torch.randperm(self.__size)
end

function KaggleDR:preprocess(img)
    return image.scale(img, 224, 224)
--[[--[
    if self.split == 'train' then
        return t.Compose{
            t.RandomSizedCrop(224),
            t.ColorJitter({
                brightness = 0.4,
                contrast = 0.4,
                saturation = 0.4,
            }),
            t.Lighting(0.1, pca.eigval, pca.eigvec),
            t.ColorNormalize(meanstd),
            t.HorizontalFlip(0.5),
        }
    elseif self.split == 'val' then
        local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
        return t.Compose{
            t.Scale(256),
            t.ColorNormalize(meanstd),
            Crop(224),
        }
    else
        error('invalid split: ' .. self.split)
    end
--]]--]
end

function KaggleDR:getSplitName()
    return self.split
end


return M.KaggleDR
