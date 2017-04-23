require 'hdf5'
require 'optim'
require 'pl'
require 'paths'
require 'image'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'nngraph'
require 'cudnn'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
local lfwHd5 = hdf5.open('datasets/lfw.hdf5', 'r')
local data = lfwHd5:read('lfw'):all()
data:mul(2):add(-1)
print(data:size(1), data:size(2), data:size(3))
lfwHd5:close()
