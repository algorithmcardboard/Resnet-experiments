require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt)
    local model = require('models/resnet')(opt)

    cudnn.fastest = true
    cudnn.benchmark = true

    local criterion = nn.MSECriterion():cuda()

    return model, criterion
end

return M
