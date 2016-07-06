local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('Kaggle Diabetic Retinopathy detection using residual networks')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-data',       '',         'Path to dataset')
    cmd:option('-trainL',         '',         'Path to train Labels')
    cmd:option('-testL',         '',         'Path to test Labels')
    cmd:option('-val',         '10',         'Percentage to use for validation set')
    cmd:option('-dataP',         '20',         'Percentage of data to use')

    local opt = cmd:parse(arg or {})

    if opt.dataP then
        opt.dataP = tonumber(opt.dataP)
    end

    if opt.val then
        opt.val = tonumber(opt.val)
    end

    --[[
    if opt.data == '' or not paths.dirp(opt.data) then
        cmd:error('Invalid data path ' .. opt.data)
    end
    --]]

    return opt
end

return M
