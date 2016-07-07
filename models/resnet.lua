local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
    local depth = opt.depth
    local shortCutType = opt.shortCutType or 'B'
    local iChannels

    local function shortcut(nInputPlane, nOutputPlane, stride)
        local useConv = shortCutType == 'C' or
            (shortCutType == 'B' and nInputPlane ~= nOutputPlane)

        if useConv then
            return nn.Sequential()
                :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
                :add(SBatchNorm(nOutputPlane))
        elseif nInputPlane ~= nOutputPlane then
            return nn.Sequential()
                :add(nn.SpatialAveragePooling(1, 1, stride, stride))
                :add(nn.concat(2)
                    :add(nn.Identity())
                    :add(nn.MulConstant(0)))
        else
            return nn.Identity()
        end
    end


end

return createModel
