local stringx = require('pl.stringx')
local cmd = torch.CmdLine()
cmd:option("-beam_size",7,"beam_size")
cmd:option("-batch_size",128,"decoding batch_size")
cmd:option("-dimension",512,"vector dimensionality")
cmd:option("-params_file","params","")
cmd:option("-model_file","model1","")
cmd:option("-setting","BS","setting for decoding, sampling, BS, DiverseBS,StochasticGreedy")
cmd:option("-DiverseRate",0,"")
cmd:option("-InputFile","../data/t_given_s_test.txt","")
cmd:option("-OutputFile","output.txt","")
cmd:option("-max_length",50,"")
cmd:option("-min_length",0,"")
cmd:option("-NBest",false,"output N-best list or just a simple output")
-- cmd:option("-gpu_index",1,"the index of GPU to use")
cmd:option("-allowUNK",false,"whether allowing to generate UNK")
cmd:option("-MMI",false,"")
cmd:option("-onlyPred",true,"")
cmd:option("-MMI_params_file","../atten/save_s_given_t/params","")
cmd:option("-MMI_model_file","","")
cmd:option("-max_decoded_num",0,"")
cmd:option("-output_source_target_side_by_side",true,"")
cmd:option("-StochasticGreedyNum",1,"")
cmd:option("-target_length",0,"force the length of the generated target, 0 means there is no such constraints")
cmd:option("-dictPath","../data/movie_25000","")
cmd:option("-PrintOutIllustrationSample",false,"")
cmd:option("-model_file", "save/model1")

local params= cmd:parse(arg)
print(params)
return params;
