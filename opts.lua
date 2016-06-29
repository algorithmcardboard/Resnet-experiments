local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('Kaggle Diabetic Retinopathy detection using residual networks')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-data',       '',         'Path to dataset')

    local opt = cmd:parse(arg or {})
    return opt
end

return M
