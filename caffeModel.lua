require 'nn'
require 'cunn'
require 'optim'
require 'loadcaffe'
require 'cudnn'
--[[
   1. Load Model
   2. Replace the last fc layer / Change learning rate accordingly
   3. Add criterion
   4. Convert model to CUDA
]]--

-- 1. Load Model

local binary = opt.pretrainedModelPath
local prototxt = opt.pretrainedPrototxtPath
model = loadcaffe.load(prototxt, binary, 'cudnn') -- Note: nn doesn't support groups, use cudnn backend

print('=> Model')
print(model)
print('=> Criterion')
print(criterion)

-- 2. Replace the last fc layer / Change learning rate accordingly
-- Divide into two groups
model_new = nn.Sequential()
model_new1 = nn.Sequential()
model_new2 = nn.Sequential()
count = 0
for i, module in ipairs(model.modules) do
   count = count + 1
end
for i, module in ipairs(model.modules) do
   if(i == count - 1) then
      print(module)
      local nClasses = opt.nClasses
      model_new2:add(nn.Linear(4096, nClasses))
      model_new2:add(cudnn.LogSoftMax())
      break
   else
      model_new1:add(module)
   end
end
model_new:add(model_new1):add(model_new2)
model = model_new
print('=> New Model')
print(model)

-- 3. Add criterion

print('=> Criterion')
criterion = nn.ClassNLLCriterion()
print(criterion)

-- 4. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
model = model:cuda()
criterion:cuda()
print 'successfully load caffe model'

collectgarbage()
