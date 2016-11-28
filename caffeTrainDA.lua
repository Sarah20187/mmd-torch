--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
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

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   3 * 1e-4,   5e-4, },
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

    if(opt.policy == 'step') then
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
    end

    -- todo-hy: not implemented
    -- here we implement a fake inv policy: set epochSize small enough but not too small to hinder the parallel
    -- recommend epochSize:10
    -- at the same time making the test run every 10 epochs(10*10 = 100 iters)
    if(opt.policy == 'inv') then
      if(epoch == 1) then
        local params, newRegime = paramsForEpoch(1)
        learningRateBase = params.learningRate
        optimState = {
           learningRate = params.learningRate,
           learningRateDecay = 0.0,
           momentum = opt.momentum,
           dampening = 0.0,
           weightDecay = 0.0
        }
        optimStateLastFC = {
           learningRate = params.learningRate * 10,
           learningRateDecay = 0.0,
           momentum = opt.momentum,
           dampening = 0.0,
           weightDecay = 0.0
        }      
      else
        -- todo-hy: perhaps the decay it too small?
        optimState.learningRate = learningRateBase * torch.pow((1 + opt.gamma * epoch * 10), (-1  * opt.power))
        optimStateLastFC.learningRate = optimState.learningRate * 10
        -- print(optimState.learningRate)
      end
    end
    print('current lr: ')
    print(optimState.learningRate)
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
            local inputsSource, labelsSource = trainLoader:sample(opt.batchSize)
            local inputsTarget = trainTargetLoader:sampleTarget(opt.batchSize)
            -- print('successfully load inputsTarget: ')
            -- print(inputsTarget:size())
            return inputsSource, inputsTarget, labelsSource 
         end,
         -- the end callback (runs in the main thread)
         trainBatchMMD
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
    -- saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
    -- torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
-- else it would cost a huge time
local inputsSource = torch.CudaTensor()
local labelsSource = torch.CudaTensor()
local inputsTarget = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters2, gradParameters2 = model.modules[2]:getParameters()
local parameters1, gradParameters1 = model.modules[1]:getParameters() 

-- 5 train with MMD
function trainBatchMMD(inputsSourceCPU, inputsTargetCPU, labelsSourceCPU)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()
    -- transfer over to GPU
    inputsSource:resize(inputsSourceCPU:size()):copy(inputsSourceCPU)
    labelsSource:resize(labelsSourceCPU:size()):copy(labelsSourceCPU)
    inputsTarget:resize(inputsTargetCPU:size()):copy(inputsTargetCPU)

    -- Update separately 
    local err, outputs
    model:zeroGradParameters()
    for i, module_ in ipairs(model.modules) do
      if(i == 1) then
        outputs1 = module_:forward({inputsSource, inputsTarget})
      end
      if(i == 2) then
        outputs2 = module_:forward(outputs1)
      end
      if(i == 3) then
        outputs3 = module_:forward(outputs2)
      end
    end
    err = criterion:forward(outputs3[1], labelsSource)
    err_mmd_fc7 = criterionMMD_fc7:forward(outputs1)
    err_mmd_fc8 = criterionMMD_fc8:forward(outputs2)
    print("err_mmd_fc7 err: ".. tostring(err_mmd_fc7))
    print("err_mmd_fc8 err: ".. tostring(err_mmd_fc8))

    local gradOutputs = criterion:backward(outputs3[1], labelsSource)   
    local zeros = torch.CudaTensor()
    zeros:resize(gradOutputs:size())
    zeros:zero()
    feval = function(x)
      gradOutputs = model.modules[3]:backward(outputs2, {gradOutputs, zeros})
      gradOutputs = model.modules[2]:backward(outputs1, gradOutputs)
      return err, gradParameters2
    end   
    optim.sgd(feval, parameters2, optimStateLastFC)  
    feval = function(x)
      model.modules[1]:backward({inputsSource, inputsTarget}, gradOutputs)
      return err, gradParameters1
    end 
    optim.sgd(feval, parameters1, optimState)


    -- mmd: fc8
    gradOutputs_mmd_fc8 = criterionMMD_fc8:backward(outputs2)
    feval = function(x)
      gradOutputs = model.modules[2]:backward(outputs1, gradOutputs_mmd_fc8)
      return err, gradParameters2
    end   
    optim.sgd(feval, parameters2, optimStateLastFC)  
    feval = function(x)
      model.modules[1]:backward({inputsSource, inputsTarget}, gradOutputs)
      return err, gradParameters1
    end 
    optim.sgd(feval, parameters1, optimState)

    -- mmd: fc7
    gradOutputs_mmd_fc7 = criterionMMD_fc7:backward(outputs1)
    feval = function(x)
      model.modules[1]:backward({inputsSource, inputsTarget}, gradOutputs_mmd_fc7)
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
        local _,prediction_sorted = outputs3[1]:float():sort(2, true) -- descending
        for i=1,opt.batchSize do
    if prediction_sorted[i][1] == labelsSourceCPU[i] then
        top1_epoch = top1_epoch + 1;
        top1 = top1 + 1
    end
        end
        top1 = top1 * 100 / opt.batchSize;
    end
    -- Calculate top-1 error, and print information
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.6f DataLoadingTime %.3f'):format(
           epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
           optimState.learningRate, dataLoadingTime))

    dataTimer:reset()
end
