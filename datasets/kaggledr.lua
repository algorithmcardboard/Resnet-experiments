local image = require 'image';
local t = require 'datasets/transforms'

local M = {}

local KaggleDR = torch.class('eyeart.KaggleDR', M)

local DR_LEVELS = {0, 1, 2, 3, 4}


function KaggleDR:__init(opt, split, info)
    self.split = split
    self.dir = opt.data
    self.info = info
    self.data = info.data
    if split == 'train' then
        self.classDistribution = info.classDistribution
    end
    self.__size = info.size
    self.R = 0.975

    self.initialWeights = torch.Tensor({1.36, 14.4, 6.64, 40.2, 49.6})
    -- self.finalWeights = torch.Tensor({1.36, 14.4, 6.64, 40.2, 49.6})
    self.finalWeights = torch.Tensor({1, 2, 2, 2, 2})

    self.indices = torch.linspace(1, self.data:size(1), self.data:size(1))

    self.classIndices = {}

    for i, level in ipairs(DR_LEVELS) do
        self.classIndices[level] = self.indices[self.data[{{}, {3}}]:eq(level)]
    end

    print('classIndices are ', split, self.classIndices, self.data:size())
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
    local perm = torch.randperm(self.__size):long()
    if self.split ~= 'train' then
        -- print('returning non train permutation "'.. self.split .. '"')
        return perm
    end

    -- do this only for training set
    --
    local weights = self.R^(epoch -1) * self.initialWeights + (1 - self.R^(epoch - 1)) * self.finalWeights
    weights = torch.cmul(weights, self.classDistribution)

    dr_levels = torch.histc(torch.multinomial(weights, self.__size, true):double(), 5, 1, 5):totable()
    -- print(dr_levels)
    local indices
    for level, count in pairs(dr_levels) do
        local curIndices = self.classIndices[level -1]
        local shuffle = torch.randperm(curIndices:size(1)):long()
        curIndices = curIndices:index(1, shuffle)
        local mul_factor = math.ceil(count/curIndices:size(1))
        -- print('mul_factor is ', mul_factor)

        if not indices then
            indices = curIndices:repeatTensor(mul_factor)[{{1,count}}]
            -- print('current indices size is ', indices:size())
        else
            indices = torch.cat(indices, curIndices:repeatTensor(mul_factor)[{{1,count}}], 1)
        end
    end
    -- print('final indices size is ', indices:size())

    return indices:index(1, perm)
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
