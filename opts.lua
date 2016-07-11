local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('Kaggle Diabetic Retinopathy detection using residual networks')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-data',             '',             'Path to dataset')
    cmd:option('-trainL',           '',             'Path to train Labels')
    cmd:option('-testL',            '',             'Path to test Labels')
    cmd:option('-val',              10,             'Percentage to use for validation set')
    cmd:option('-dataP',            20,             'Percentage of data to use')
    cmd:option('-nEpochs',          300,            'Maximum epochs')
    cmd:option('-batchSize',        64,             'Batch size for epochs')
    cmd:option('-nThreads',         6,              'Number of dataloading threads')
    cmd:option('-manualSeed',       0,              'Manually set RNG seed')
    cmd:option('-depth',            34,             'Depth of model')
    cmd:option('-LR',               0.1,            'initial learning rate')
    cmd:option('-momentum',         0.9,            'momentum')
    cmd:option('-weightDecay',      1e-4,           'weight decay')
    cmd:option('-logDir',           'logs',         'log directory')

    local opt = cmd:parse(arg or {})

    if opt.data == '' or not paths.dirp(opt.data) then
        cmd:error('Invalid data path ' .. opt.data)
    end

    return opt
end

return M
