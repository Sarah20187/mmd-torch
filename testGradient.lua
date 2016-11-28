require 'torch'
require 'nn'
require 'mmd'


module_ = nn.mmdCriterion()
input1 = torch.rand(8, 2)
input2 = torch.rand(8, 2)
for i=1, 8 do
    for j=1, 2 do
        input1[i][j] = (i-1) * 2 + j-1
        input2[i][j] = input1[i][j] + 1
    end
end

err = module_:forward({input1, input2})

err2 = module_:backward({input1, input2})


-- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- --

require 'torch'
require 'nn'
require 'totem'
require 'nngraph'
require 'mmd'

test = {}
tester = totem.Tester()

function test.test_grad()
    local input3 = torch.randn(7)
    local prediction = nn.Identity()()
    local zero = nn.MulConstant(0)(prediction)
    local target = nn.AddConstant(0.023)(zero)
    -- local mse = nn.MSECriterion()({prediction, target})
    local mse = nn.AngleCriterion()({prediction, target})
    local net = nn.gModule({prediction}, {mse})
    totem.nn.checkGradients(tester, net, input3)
end

tester:add(test):run()
