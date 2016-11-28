--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')
if opt.caffe == 1 then
    if opt.isDA == 1 then
        paths.dofile('caffeModelDA.lua')
    else
        paths.dofile('caffeModel.lua')
    end
else
    paths.dofile('model.lua')
end

opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
if opt.caffe == 1 then
    if opt.isDA == 1 then
        paths.dofile('caffeTrainDA.lua')
    else
        paths.dofile('caffeTrain.lua')
    end
else
    paths.dofile('train.lua')
end

epoch = -1
if opt.caffe == 1 and opt.isDA == 1 then
    paths.dofile('caffeTestDA.lua')
else
    paths.dofile('test.lua')
end

-- test()

epoch = opt.epochNumber
-- kinda ugly
if opt.caffe == 1 and opt.isDA == 1 and opt.policy == 'inv' then
    for i=1,opt.nEpochs do
        train()
        if(epoch % 10 == 0) then test() end
        epoch = epoch + 1
    end
else
    for i=1,opt.nEpochs do
    train()
    test()
    epoch = epoch + 1
    end
end


