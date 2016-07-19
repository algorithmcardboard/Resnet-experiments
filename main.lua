require 'torch';
require 'cutorch';
require 'optim';
require 'os';

-- torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
torch.manualSeed(0)
cutorch.manualSeedAll(0)

local opts = require 'opts'
local opt = opts.parse(arg)

local dateT = os.date('*t')

local dateStr = dateT.year .. dateT.month .. dateT.day .. '_' .. dateT.hour .. dateT.min .. dateT.sec

local logger = optim.Logger(opt.logDir .. '/resnet_' .. dateStr .. '.log' .. (opt.name and '.' .. opt.name or ''))
logger:setNames{'train_loss', 'train_mse', 'val_loss', 'val_mse'}
logger:style{'+-', '+-', '+-', '+-'}

local statsLogger = optim.Logger(opt.logDir ..'/stats.log')
statsLogger:setNames{'CPU_MEMORY', 'CPU1_TEMPERATURE', 'CPU2_TEMPERATURE', 'CPU3_TEMPERATURE', 'CPU4_TEMPERATURE', 'GPU_TEMPERATURE', 'GPU_MEMORY', 'GPU_UTILIZATION'}


local models = require 'models/init'
local DataLoader = require 'dataloader'
local Trainer = require 'train'

local model, criterion = models.setup(opt)

local valLoader, trainLoader = DataLoader.create(opt, {'train'})

print('trainLoader ', trainLoader)
print('valLoader ', valLoader)

local trainer = Trainer(model, criterion, opt, optimState)

local function get_cpu_temperatures()
    local handle = io.popen("sensors | tail -n5 |  awk -F ' ' '{print $3}' | tr '\n' ' '")
    local cpu_temperature = handle:read('*a'):split(' ')
    cpu_temperature[#cpu_temperature] = nil

    for i, v in ipairs(cpu_temperature) do
        cpu_temperature[i] = cpu_temperature[i]:match(".(%d+%.%d+)")
    end

    return cpu_temperature
end

local function get_gpu_stats()
    local handle = io.popen("nvidia-smi -q -d temperature | grep Current")
    local result = handle:read('*a')
    local temperature = (result:split(':')[2]:gsub("%s", "")):match("(%d*).*")


    handle = io.popen("nvidia-smi -q -d memory | grep Used | head -n 1")
    result = handle:read('*a')
    local memory = (result:split(':')[2]:gsub("%s", "")):match("(%d*).*")

    handle = io.popen("nvidia-smi -q -d utilization | grep Gpu")
    result = handle:read('*a')
    local utilization = (result:split(':')[2]:gsub("%s", "")):match("(%d*).*")

    return temperature, memory, utilization
end

local function logHardwareStatistics()
    local cpu_memory = collectgarbage('count')/1024

    local cpu_temperature = get_cpu_temperatures()

    local gpu_temp, gpu_mem, gpu_util = get_gpu_stats()

    statsLogger:add{cpu_memory, cpu_temperature[1], cpu_temperature[2], cpu_temperature[3], cpu_temperature[4], gpu_temp, gpu_mem, gpu_util}
end

for epoch = 1, opt.nEpochs do
    local train_loss, train_mse = trainer:train(epoch, trainLoader)
    local val_loss, val_mse = trainer:validate(epoch, valLoader)

    print("Scores are ", train_loss, train_mse, val_loss, val_mse);

    -- code to sum/do operations with previous valStastistics and vs
    logHardwareStatistics()

    logger:add{train_loss, train_mse, val_loss, val_mse}
    logger:plot()
end

print('done with training')
