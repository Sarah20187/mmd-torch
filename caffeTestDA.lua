--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local top1_center, loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   loss = 0
   for i = 1, math.ceil(nTest/opt.batchSize) do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = math.min(nTest, indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputsTarget, labelsTarget = testLoader:get(indexStart, indexEnd)
            local zeros = torch.Tensor()
            zeros:resize(inputsTarget:size())
            zeros:zero()
            return inputsTarget, zeros, labelsTarget
         end,
         -- callback that is run in the main thread once the work is done
         testBatchDA
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   loss = loss / nTest -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputsSource = torch.CudaTensor()
local inputsTarget = torch.CudaTensor()
local labels = torch.CudaTensor()

-- Note though we have two separate softmax output, the thread of inputsSource will be used for inputsTarget
-- Hence: when using this function: 
-- set inputsSourceCPU = inputsTargetCPU
-- set inputsTargetCPU = zeros
-- set labelsSourceCPU = targetLabels
function testBatchDA(inputsSourceCPU, inputsTargetCPU, labelsSourceCPU) 
   batchNumber = batchNumber + opt.batchSize

   inputsSource:resize(inputsSourceCPU:size()):copy(inputsSourceCPU)
   inputsTarget:resize(inputsTargetCPU:size()):copy(inputsTargetCPU)
   labels:resize(labelsSourceCPU:size()):copy(labelsSourceCPU)

   inputs = {inputsSource, inputsTarget}
   local outputs = model:forward(inputs)[1]
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()
   local pred = outputs:float()

   loss = loss + err * outputs:size(1)

   local _, pred_sorted = pred:sort(2, true)
   for i=1,pred:size(1) do
      local g = labelsSourceCPU[i]
      if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
   end
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   end
end

