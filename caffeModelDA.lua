require 'nn'
require 'cunn'
require 'optim'
require 'loadcaffe'
require 'cudnn'
require 'mmd'


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
      for k,v in ipairs(model_new2:findModules('nn.Linear')) do
         v.weight:normal(0,0.01)
         v.bias:zero()
      end
      break
   else
      model_new1:add(module)
   end
end

map = nn.MapTable()
map:add(model_new1)
local model_new2_2 = model_new2:clone() -- branch 2
for k,v in ipairs(model_new2_2:findModules('nn.Linear')) do
   v.weight:normal(0,0.01)
   v.bias:zero()
end
parallel = nn.ParallelTable()
parallel:add(model_new2):add(model_new2_2) -- which I am doubtful.. shouldn't it be weight sharing?
parallel_softmax = nn.ParallelTable()
parallel_softmax:add(cudnn.LogSoftMax()):add(cudnn.LogSoftMax())

model_new:add(map):add(parallel):add(parallel_softmax)
model = model_new
print('=> New Model')
print(model)

-- 3. Add criterion

print('=> Criterion')
criterion = nn.ClassNLLCriterion()
print(criterion)

print('=> MMD Criterion')
criterionMMD_fc7 = nn.mmdCriterion()
print(criterionMMD_fc7)
criterionMMD_fc8 = nn.mmdCriterion()
print(criterionMMD_fc8)
-- 4. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
model = model:cuda()
criterion:cuda()
criterionMMD_fc7:cuda()
criterionMMD_fc8:cuda()
print 'successfully load caffe model'

collectgarbage()
