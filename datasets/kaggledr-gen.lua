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


local function _pruneData(imageLabels, classDistribution, dataPercentage)

    local indices = torch.linspace(1, imageLabels:size(1), imageLabels:size(1)):long()
    local unSymmetricEyes = imageLabels:index(1, indices[imageLabels[{{}, {2}}]:ne(imageLabels[{{}, {3}}])])

    assert(unSymmetricEyes[{{}, {2}}]:eq(unSymmetricEyes[{{}, {3}}]):sum() == 0, 'Not all unSymmetric eyes')

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

    for i, v in pairs(toTruncate) do
        print('in toTruncate loop', v, math.ceil(classDistribution[i] * dataPercentage * 0.01))
        if v > 0 then
            print('Truncating for class ', i)

            local l_eq_val, r_eq_val = unSymmetricEyes[{{}, {2}}]:eq(i), unSymmetricEyes[{{}, {3}}]:eq(i)
            local unSymIndices = (l_eq_val + r_eq_val):reshape(unSymmetricEyes:size(1))

            print('total available in class '.. i .. ' ' .. unSymIndices:sum())
            assert(unSymIndices:eq(2):sum() == 0, 'Unsymmetric eyes not in dataset')
            assert(unSymIndices:sum() == (l_eq_val:sum() + r_eq_val:sum()), 'Some problem with unsymmetric eyes')

            local indices = torch.linspace(1, unSymmetricEyes:size(1), unSymmetricEyes:size(1)):long()

            local length = unSymIndices:sum()
            indices = (indices[unSymIndices])[{{1, length - v}}]

            print('before sum is ', unSymIndices:sum())
            unSymIndices:indexFill(1, indices, 0)
            print('after sum is ', unSymIndices:sum())

            unSymIndices = unSymIndices:ne(1)
            print(unSymIndices:sum(), unSymmetricEyes:size(1), unSymIndices:size(1))

            assert(unSymIndices:size(1) == unSymmetricEyes:size(1), 'unequal sizes')


            indices = torch.linspace(1, unSymmetricEyes:size(1), unSymmetricEyes:size(1)):long()
            unSymmetricEyes = unSymmetricEyes:index(1, indices[unSymIndices])
            print('unSymmetricEyes size is ')
            print(unSymmetricEyes:size())
        end
    end

    return imageLabels
end

local function _processRawData(split, labelFile, headers, opt)

    local imageLabels;
    if paths.filep('imageLabels.t7') then
        print('reading from t7 file');
        imageLabels = torch.load('imageLabels.t7')
    else
        print('processing csv');
        imageLabels = _readCSVToTensor(labelFile, headers, opt.data)
    end

    local data = {}
    data.classDistribution = {}

    for i = 0, 4 do
        data.classDistribution[i] = imageLabels[{{}, {2}}]:eq(i):sum() + imageLabels[{{}, {3}}]:eq(i):sum()
    end

    imageLabels = _pruneData(imageLabels, data.classDistribution, opt.dataP)

    local indices = torch.linspace(1, imageLabels:size(1), imageLabels:size(1)):long()
    local classToImages = {}
    local dataSize = 0
    local info = {}

    for i = 0,4 do
        local selected = indices[imageLabels[{{}, {3}}]:eq(i)]
        classToImages[i] = imageLabels:index(1,selected)
    end

    if opt.dataP > 0 and opt.dataP <= 100 and opt.dataP%1 == 0 then
        classToImages, dataSize = _pruneData(classToImages, opt.dataP)
    end

    if split=="train" and opt.val > 0 and opt.val < 100 and opt.val % 1 == 0 then
        classToImages, dataSize, validationImages, valSize = _train_validation_split(classToImages, opt.val)
        info['val'] = {}
        info.val['classToImages'] = validationImages 
        info.val['size'] = valSize
        info.val['data'] = flatten(validationImages, valSize)
    end

    info[split] = {}
    info[split]['classToImages'] = classToImages
    info[split]['size'] = dataSize
    info[split]['data'] = flatten(classToImages, dataSize)

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
