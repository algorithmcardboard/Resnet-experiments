local image = require 'image';

local M = {}

local KaggleDR = torch.class('eyeart.KaggleDR', M)

function KaggleDR.create(opt, setName)
    local labelFile = opt[setName..'L']

    assert(paths.dirp(opt.data), 'data folder is not valid'..opt.data)
    assert(paths.filep(labelFile), 'opt.'..setName..'L is not valid'..labelFile)

    local fileName = KaggleDR.saveLabelFile(setName, labelFile, true, opt)

    return M.KaggleDR(opt, setName, fileName)
end

function KaggleDR.pruneData(allData, percentage)
    print('in prune data function ', percentage)
    for dr_level = 0,4 do
        local size = allData[dr_level]:size(1)
        local perm = torch.randperm(size):long()
        local targetSize = math.ceil(size * percentage/100)
        local indices = torch.linspace(1, targetSize, targetSize):long()

        print('size of ' .. dr_level .. ' is ' .. size)

        allData[dr_level] = allData[dr_level]:index(1, perm)
        allData[dr_level] = allData[dr_level]:index(1, indices)
    end

    return allData;
end

function KaggleDR.saveLabelFile(split, labelFile, headers, opt)
    local imageLabels = torch.DoubleTensor(35126,3)

    local skip = headers or true
    local count = 1
    for line in io.lines(labelFile) do
        if not skip then
            local image, pos, dr_level = line:match("(%d*)_(%a*),(%d*)")

            pos_int = 2
            if pos == "left" then
                pos_int = 1
            end

            imageLabels[count][1] = image
            imageLabels[count][2] = pos_int
            imageLabels[count][3] = dr_level
            count = count + 1
        end
        skip = false
    end

    local perm = torch.randperm(imageLabels:size(1))
    imageLabels = imageLabels:index(1, perm:long())
    
    local indices = torch.linspace(1, imageLabels:size(1), imageLabels:size(1)):long()
    local classToImages = {}

    for i = 0,4 do
        local selected = indices[imageLabels[{{}, {3}}]:eq(i)]
        classToImages[i] = imageLabels:index(1,selected)
    end

    if opt.dataP > 0 and opt.dataP < 100 and opt.dataP%1 == 0 then
        classToImages = KaggleDR.pruneData(classToImages, opt.dataP)
    end

    local fileName = table.concat({"train", opt.dataP, opt.val, "t7"}, '.');
    torch.save(fileName, classToImages)
    return fileName
end

function KaggleDR:__init(opt, split, fileName)
    self.split = split
    self.dir = opt.data
    self.labelFile = opt[split..'L']
    self.imageLabels = torch.load(fileName)


    if self.split == 'train' and opt.val then
        print('doing validation')
    else
        print('no validation')
    end
end

function KaggleDR:size()
end

return M.KaggleDR
