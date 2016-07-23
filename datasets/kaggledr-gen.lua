local KaggleDR = require('datasets/kaggledr')

M = {}

local _DR_LEVELS = {0, 1, 2, 3, 4}

local function _getNumRowsColsFromCSV(labelFile, headers)

    local skip = headers or true
    local COLS = 0
    local ROWS = 0

    for line in io.lines(labelFile) do
        if ROWS == 0 then
            COLS = #line:split(',')
        end

        ROWS = ROWS + 1
    end
    if headers then
        ROWS = ROWS - 1
    end

    ROWS = ROWS / 2 -- we are going to go by encounters
    COLS = COLS + 4 - 1 -- one for left/right image present.  Another for left dr and right dr 

    return ROWS, COLS
end

local function _readCSVToTensor(labelFile, headers, dataDir)

    local ROWS, COLS = _getNumRowsColsFromCSV(labelFile, headers)
    local imageLabels = torch.zeros(ROWS,COLS):double()

    local encounter_ids, enc_index = {}, 1
    skip = headers or true

    for line in io.lines(labelFile) do
        if not skip then
            local image, pos, dr_level = line:match("(%d*)_(%a*),(%d*)")
            image = tonumber(image)

            pos_int = 2
            if pos == "left" then
                pos_int = 1
            end

            if encounter_ids[image] ==  nil then
                encounter_ids[image] = enc_index
                enc_index = enc_index + 1
            end

            local enc_id = encounter_ids[image]
            local file_present = 0 

            local fileName = dataDir ..'/'.. image..'_'..pos..'.jpg' 
            if paths.filep(fileName) then
                file_present = 1
            end

            if pos_int == 1 then
                image_bool = 2
                dr_pos = 4
            else
                image_bool = 3
                dr_pos = 5
            end

            -- encounter_id, left_present, right_present, left_dr, right_dr
            imageLabels[enc_id][1] = image
            imageLabels[enc_id][image_bool] = file_present
            imageLabels[enc_id][dr_pos] = dr_level
        end
        skip = false
    end

    local indices = torch.linspace(1, imageLabels:size(1), imageLabels:size(1)):long()
    imageLabels = imageLabels:index(1, indices[imageLabels[{{}, {2}}]:ne(0)]:long())

    indices = torch.linspace(1, imageLabels:size(1), imageLabels:size(1)):long()
    imageLabels = imageLabels:index(1, indices[imageLabels[{{}, {3}}]:ne(0)]:long())

    assert(imageLabels:size(1) - imageLabels[{{}, {2}}]:sum() == 0, 'All absetnt left files are not removed')
    assert(imageLabels:size(1) - imageLabels[{{}, {3}}]:sum() == 0, 'All absetnt right files are not removed')

    imageLabels = imageLabels:index(2, torch.LongTensor({1, 4, 5})):double()

    torch.save('imageLabels.t7', imageLabels)
    return imageLabels
end

local function _getUnsymmetricEyes(imageLabels)
    local indices = torch.linspace(1, imageLabels:size(1), imageLabels:size(1)):long()
    local unSymmetricEyes = imageLabels:index(1, indices[imageLabels[{{}, {2}}]:ne(imageLabels[{{}, {3}}])])

    assert(unSymmetricEyes[{{}, {2}}]:eq(unSymmetricEyes[{{}, {3}}]):sum() == 0, 'Not all unSymmetric eyes')

    return unSymmetricEyes
end

local function _get_symmetric_eyes(imageLabels)
    local indices = torch.linspace(1, imageLabels:size(1), imageLabels:size(1)):long()
    local symmetricEyes = imageLabels:index(1, indices[imageLabels[{{}, {2}}]:eq(imageLabels[{{}, {3}}])])

    assert(symmetricEyes[{{}, {2}}]:ne(symmetricEyes[{{}, {3}}]):sum() == 0, 'Not all Symmetric eyes')
    return  symmetricEyes
end

local function _truncate(imageLabels, toTruncate, classDistribution, dataPercentage)

    local unSymmetricEyes = _getUnsymmetricEyes(imageLabels)

    for i, v in pairs(toTruncate) do
        if v > 0 then
            -- print('in toTruncate loop', v, math.ceil(classDistribution[i] * dataPercentage * 0.01))

            local l_eq_val, r_eq_val = unSymmetricEyes[{{}, {2}}]:eq(i), unSymmetricEyes[{{}, {3}}]:eq(i)
            local unSymIndices = (l_eq_val + r_eq_val):reshape(unSymmetricEyes:size(1))

            -- print('total available in class '.. i .. ' ' .. unSymIndices:sum())
            assert(unSymIndices:eq(2):sum() == 0, 'Unsymmetric eyes not in dataset')
            assert(unSymIndices:sum() == (l_eq_val:sum() + r_eq_val:sum()), 'Some problem with unsymmetric eyes')

            local indices = torch.linspace(1, unSymmetricEyes:size(1), unSymmetricEyes:size(1)):long()

            local length = unSymIndices:sum()
            indices = (indices[unSymIndices])[{{1, length - v}}]
            unSymIndices:indexFill(1, indices, 0)

            indices = torch.linspace(1, unSymmetricEyes:size(1), unSymmetricEyes:size(1)):long()
            local discardedEncounters = unSymmetricEyes:index(1, indices[unSymIndices])

            -- housekeeping
            for dr_level = 0, 4 do 
                if dr_level ~= i and toTruncate[dr_level] then
                    local num_cases = discardedEncounters[{{},{2}}]:eq(dr_level):sum() + discardedEncounters[{{},{3}}]:eq(dr_level):sum()
                    toTruncate[dr_level] = toTruncate[dr_level] - num_cases
                end
            end

            unSymIndices = unSymIndices:ne(1)

            assert(unSymIndices:size(1) == unSymmetricEyes:size(1), 'unequal sizes')

            indices = torch.linspace(1, unSymmetricEyes:size(1), unSymmetricEyes:size(1)):long()
            unSymmetricEyes = unSymmetricEyes:index(1, indices[unSymIndices])

            -- print('unSymmetricEyes size is ')
            -- print(unSymmetricEyes:size())
        end
    end

    local requiredSize = math.ceil(imageLabels:size(1) * dataPercentage * 0.01) - unSymmetricEyes:size(1)
    local symmetricEyes = _get_symmetric_eyes(imageLabels)
    local perm = torch.randperm(symmetricEyes:size(1)):long()

    symmetricEyes = symmetricEyes:index(1, perm)
    symmetricEyes = symmetricEyes[{{1, requiredSize}}]

    imageLabels = torch.cat(unSymmetricEyes, symmetricEyes, 1)
    return imageLabels
end

local function _pruneData(imageLabels, classDistribution, dataPercentage)

    local unSymmetricEyes = _getUnsymmetricEyes(imageLabels)

    local toTruncate = {};

    print('dataPercentage is ', dataPercentage)

    for i, v in pairs(classDistribution) do
        local totalSamples, requiredSamples = v, math.ceil(v*dataPercentage*0.01)
        print(i, totalSamples, requiredSamples)

        local available = unSymmetricEyes[{{}, {2}}]:eq(i):sum() + unSymmetricEyes[{{}, {3}}]:eq(i):sum()

        if available > requiredSamples then
            toTruncate[i] = available - requiredSamples
        end
    end

    print('toTruncate is ', toTruncate)

    imageLabels = _truncate(imageLabels, toTruncate, classDistribution, dataPercentage)
    return imageLabels
end

local function _train_validation_split(imageLabels, valPercentage)
    local dataSize, valSize = imageLabels:size(1), math.ceil(imageLabels:size(1) * valPercentage * 0.01)
    local shuffle = torch.randperm(dataSize):long()

    imageLabels = imageLabels:index(1, shuffle)

    return imageLabels[{{1, dataSize - valSize}}], imageLabels[{{dataSize - valSize + 1, dataSize}}]
end

local function _flatten(imageLabels)
    local left_dr = imageLabels:index(2, torch.LongTensor({1, 2}))
    local right_dr = imageLabels:index(2, torch.LongTensor({1, 3}))

    right_dr = torch.cat(right_dr, torch.randn(right_dr:size(1)):fill(2):reshape(right_dr:size(1), 1), 2)
    right_dr = right_dr:index(2, torch.LongTensor({1,3,2}))

    left_dr = torch.cat(left_dr, torch.ones(left_dr:size(1)):reshape(left_dr:size(1), 1), 2)
    left_dr = left_dr:index(2, torch.LongTensor({1,3,2}))


    return torch.cat(left_dr, right_dr, 1)
end

local function _processRawData(split, labelFile, headers, opt)

    local imageLabels;
    if paths.filep('imageLabels.t7') then
        print('Reading from t7 file');
        imageLabels = torch.load('imageLabels.t7')
    else
        print('Processing csv');
        imageLabels = _readCSVToTensor(labelFile, headers, opt.data)
    end

    local classDistribution, distribution = {}, {}

    for i = 0, 4 do
        table.insert(distribution, imageLabels[{{}, {2}}]:eq(i):sum() + imageLabels[{{}, {3}}]:eq(i):sum())
        classDistribution[i] = distribution[i+1]
    end

    imageLabels = _pruneData(imageLabels, classDistribution, opt.dataP)
    local length = imageLabels:size(1)
    local shuffle = torch.randperm(length):long()

    imageLabels = imageLabels:index(1, shuffle)

    local info = {}
    info[split] = {}

    if split=="train" and opt.val > 0 and opt.val < 100 and opt.val % 1 == 0 then
        imageLabels, valLabels  = _train_validation_split(imageLabels, opt.val)
        valLabels = _flatten(valLabels)
        info['val'] = {}
        info.val['size'] = valLabels:size(1)
        info.val['data'] = valLabels
    end

    imageLabels = _flatten(imageLabels)
    info[split]['size'] = imageLabels:size(1)
    info[split]['data'] = imageLabels
    if(split == 'train') then
        info[split]['classDistribution'] = torch.Tensor(distribution)
    end

    local fileName = table.concat({"processed", split, opt.dataP, opt.val, "t7"}, '.');
    torch.save(fileName, info)
    return info 
end

function M.exec(opt, setName)
    local labelFile = opt[setName..'L']

    assert(paths.dirp(opt.data), 'data folder is not valid'..opt.data)
    assert(paths.filep(labelFile), 'opt.'..setName..'L is not valid'..labelFile)

    local dataSetInfo = _processRawData(setName, labelFile, true, opt)

    local dataSets = {}

    for split, info in pairs(dataSetInfo) do
        table.insert(dataSets, KaggleDR(opt, split, info))
    end

    return dataSets
end

return M
