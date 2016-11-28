--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

-- naturally it's unnecessary to modify the train.lua. 
-- However, we need to set different lr for the last layer. So we set two optimState here
-- Note there is no need to modify the test.lua
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end


local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   3 * 1e-3,   5e-4, },
        { 19,     29,   3 * 5e-4,   5e-4  },
        { 30,     43,   3 * 1e-4,   0 },
        { 44,     52,   3 * 5e-5,   0 },
        { 53,    1e8,   3 * 1e-6,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
      optimStateLastFC = {
         learningRate = params.learningRate * 10,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0

   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            local inputsTarget = trainTargetLoader:sampleTarget(opt.batchSize)
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end -- of for loop



   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
-- else it would cost a huge time
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters2, gradParameters2 = model.modules[2]:getParameters()
local parameters1, gradParameters1 = model.modules[1]:getParameters() 

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)

    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    -- Update separately 
    local err, outputs
    model:zeroGradParameters()
    for i, module_ in ipairs(model.modules) do
      if(i == 1) then
        outputs1 = module_:forward(inputs)
      else
        outputs2 = module_:forward(outputs1)
      end
    end
    err = criterion:forward(outputs2, labels)
    local gradOutputs = criterion:backward(outputs2, labels)
    
    feval = function(x)
      gradOutputs:cuda()
      outputs1:cuda()
      gradOutputs = model.modules[2]:backward(outputs1, gradOutputs)
      return err, gradParameters2
    end   
    optim.sgd(feval, parameters2, optimStateLastFC)  

    
    feval = function(x)
      gradOutputs:cuda()
      gradOutputs = model.modules[1]:backward(inputs, gradOutputs)
      return err, gradParameters1
    end 
    optim.sgd(feval, parameters1, optimState)
    -- end of Update separately

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    loss_epoch = loss_epoch + err
    -- top-1 error
    local top1 = 0
    do
        local _,prediction_sorted = outputs2:float():sort(2, true) -- descending
        for i=1,opt.batchSize do
    if prediction_sorted[i][1] == labelsCPU[i] then
        top1_epoch = top1_epoch + 1;
        top1 = top1 + 1
    end
        end
        top1 = top1 * 100 / opt.batchSize;
    end
    -- Calculate top-1 error, and print information
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
           epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
           optimState.learningRate, dataLoadingTime))

    dataTimer:reset()
end
