import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))   # 繝代Λ繝｡繝ｼ繧ｿ縺ｮ蜈ｨ繝√Ε繝ｳ繝阪Ν繧剃ｸ諡ｬ螟画鋤
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))  # 繝代Λ繝｡繝ｼ繧ｿ縺ｮ蜈ｨ繝√Ε繝ｳ繝阪Ν繧剃ｸ諡ｬ螟画鋤
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
    
def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class MultiRNN(nn.Module):
    """
    Container module for multiple stacked RNN layers.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should 
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(MultiRNN, self).__init__()

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout, 
                                         batch_first=True, bidirectional=bidirectional)
        
        

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(self, input):
        hidden = self.init_hidden(input.size(0))
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_())

class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.
    
    args:
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should 
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """
    
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(FCLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None
            
        self.init_hidden()
    
    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)
              
    def init_hidden(self):
        initrange = 1. / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)

""" D-1 Conv """
class DepthConv1d(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel, padding, dilation=1, skip=True, causal=False):
        """
        Parameters
        ----------
        input_dim:蜈･蜉帙・谺｡蜈・焚
        hidden_dim:髫繧悟ｱ､縺ｮ谺｡蜈・焚
        kernel:繧ｫ繝ｼ繝阪Ν繧ｵ繧､繧ｺ
        padding:繝代ユ繧｣繝ｳ繧ｰ驥・
        dilation:繝繧､繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ縺ｮ驥・
        skip:skip connection譛臥┌
        causal
        """
        super(DepthConv1d, self).__init__()

        self.causal = causal
        self.skip = skip        # skip connection縺ｮ譛臥┌
        
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, 1)
        """ 繝代ユ繧｣繝ｳ繧ｰ驥上・隱ｿ謨ｴ """
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding

        self.dconv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel, dilation=dilation,
                                 groups=hidden_dim, padding=self.padding)
        self.res_out = nn.Conv1d(hidden_dim, input_dim, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_dim, eps=1e-08)
            self.reg2 = cLN(hidden_dim, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_dim, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_dim, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_dim, input_dim, 1)

    def forward(self, input_data:torch.Tensor) -> torch.Tensor:
        """
        D-1Conv縺ｮ蜍穂ｽ懊ｒ螳夂ｾｩ

        Parameters
        ----------
        input_data:蜈･蜉帙ョ繝ｼ繧ｿ

        Returns
        -------
        residual:蜃ｺ蜉・谿狗蕗迚ｩ)
        """
        # print('\nDepthConv1d')
        output = self.reg1(self.nonlinearity1(self.conv1d(input_data)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))

        residual = self.res_out(output)

        if self.skip:
            skip = self.skip_out(output)
            # print('DepthConv2d_F\n')
            return residual, skip
        else:
            return residual
""" TCN """
class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False, dilated=True):
        super(TCN, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        """ normalization """
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        """ TCN for feature extraction """
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, padding=2 ** i, dilation=2 ** i, skip=skip,
                                                causal=causal))
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, padding=1, dilation=1, skip=skip, causal=causal))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)

        # print(f"Receptive field: {self.receptive_field:3d} frames.")

        """ output layer """
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))
        self.skip = skip

    def forward(self, input_data:torch.Tensor)->torch.Tensor:
        """
        TCN縺ｮ蜍穂ｽ懊ｒ螳夂ｾｩ

        Parameters
        ----------
        input_data:蜈･蜉帙ョ繝ｼ繧ｿ

        Returns
        -------
        output:蜃ｺ蜉帙ョ繝ｼ繧ｿ
        """
        # input shape: (B, N, L)

        """normalization"""
        output = self.BN(self.LN(input_data))

        """pass to TCN"""
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        """output layer"""
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output

""" Conv-TasNet繧ｯ繝ｩ繧ｹ縺ｮ螳夂ｾｩ """
class enhance_ConvTasNet(nn.Module):    # 髻ｳ貅仙ｼｷ隱ｿ
    def __init__(self, encoder_dim=512, feature_dim=128, sampling_rate=16000, win=4, layer=8, stack=3,
                 kernel=3, num_speaker=1, causal=False):    #num_speaker=1繧ゅ→繧ゅ→縺ｮ繧・▽
        super(enhance_ConvTasNet, self).__init__()

        """hyper parameters"""
        self.num_speaker = num_speaker              # 隧ｱ閠・焚
        self.encoder_dim = encoder_dim              # 繧ｨ繝ｳ繧ｳ繝ｼ繝縺ｫ蜈･蜉帙☆繧区ｬ｡蜈・焚
        self.feature_dim = feature_dim              # 迚ｹ蠕ｴ谺｡蜈・焚 (繧ｨ繝ｳ繧ｳ繝ｼ繝縺ｮ蜃ｺ蜉・
        self.win = int(sampling_rate * win / 1000)  # 遯馴聞
        self.stride = self.win // 2                 # 逡ｳ縺ｿ霎ｼ縺ｿ蜃ｦ逅・↓縺翫￠繧九ヵ繧｣繝ｫ繧ｿ縺檎ｧｻ蜍輔☆繧句ｹ・
        self.layer = layer                          # 螻､謨ｰ
        self.stack = stack                          # 繧ｹ繧ｿ繝・け謨ｰ
        self.kernel = kernel                        # 繧ｫ繝ｼ繝阪Ν
        self.causal = causal                        #
        self.channel = 1                            # 繝√Ε繝ｳ繝阪Ν謨ｰ

        """input encoder"""
        self.encoder = nn.Conv1d(in_channels=self.channel,      # 蜈･蜉帙ョ繝ｼ繧ｿ縺ｮ谺｡蜈・焚 #=1繧ゅ→繧ゅ→縺ｮ繧・▽
                                 out_channels=self.encoder_dim, # 蜃ｺ蜉帙ョ繝ｼ繧ｿ縺ｮ谺｡蜈・焚
                                 kernel_size=self.win,          # 逡ｳ縺ｿ霎ｼ縺ｿ縺ｮ繧ｵ繧､繧ｺ(豕｢蠖｢鬆伜沺縺ｪ縺ｮ縺ｧ遯馴聞縺ｪ縺ｮ?)
                                 bias=False,                    # 繝舌う繧｢繧ｹ縺ｮ譛臥┌(蜃ｺ蜉帙↓蟄ｦ鄙貞庄閭ｽ縺ｪ繝舌う繧｢繧ｹ縺ｮ霑ｽ蜉)
                                 stride=self.stride)            # 逡ｳ縺ｿ霎ｼ縺ｿ蜃ｦ逅・・遘ｻ蜍募ｹ・

        """TCN separator"""
        self.TCN = TCN(input_dim=self.encoder_dim,                       # 蜈･蜉帙ョ繝ｼ繧ｿ縺ｮ谺｡蜈・焚
                       output_dim=self.encoder_dim * self.num_speaker,   # 蜃ｺ蜉帙ョ繝ｼ繧ｿ縺ｮ谺｡蜈・焚
                       BN_dim=self.feature_dim,                          # 繝懊ヨ繝ｫ繝阪ャ繧ｯ螻､縺ｮ迚ｹ蠕ｴ谺｡蜈・焚
                       hidden_dim=self.feature_dim * 4,
                       layer=self.layer,                                 # 螻､謨ｰ
                       stack=self.stack,                                 # 繧ｹ繧ｿ繝・け謨ｰ
                       kernel=self.kernel,                               # 繧ｫ繝ｼ繝阪Ν繧ｵ繧､繧ｺ
                       causal=self.causal)
        self.receptive_field = self.TCN.receptive_field

        """output decoder"""
        self.decoder = nn.ConvTranspose1d(in_channels = self.encoder_dim,     # 蜈･蜉帶ｬ｡蜈・焚
                                          out_channels=1,                    # 蜃ｺ蜉帶ｬ｡蜈・焚 1繧ゅ→繧ゅ→縺ｮ繧・▽
                                          kernel_size= self.win,             # 繧ｫ繝ｼ繝阪Ν繧ｵ繧､繧ｺ
                                          bias=False,
                                          stride=self.stride)   # 逡ｳ縺ｿ霎ｼ縺ｿ蜃ｦ逅・・遘ｻ蜍募ｹ・

    def patting_signal(self, input_data: torch.Tensor) -> tuple:
        """
        蜈･蜉帙ョ繝ｼ繧ｿ繧偵ヱ繝・ぅ繝ｳ繧ｰ竊堤糞縺ｿ霎ｼ縺ｿ蜑阪・谺｡蜈・焚縺ｨ逡ｳ縺ｿ霎ｼ縺ｿ蠕後・谺｡蜈・焚繧貞酔縺倥↓縺吶ｋ縺溘ａ縺ｫ蜈･蜉帙ョ繝ｼ繧ｿ繧・縺ｧ蝗ｲ繧謫堺ｽ・

        Parameters
        ----------
        input_data(tensor[1,繝√Ε繝ｳ繝阪Ν謨ｰ,髻ｳ螢ｰ髟ｷ]):蜈･蜉帙ョ繝ｼ繧ｿ

        Returns
        -------
        output(tensor):蜃ｺ蜉・
        rest:
        """
        # input is the waveforms: (B, T) or (B, 1, T)
        """reshape and padding"""
        if input_data.dim() not in [2, 3]:   # input縺ｮ谺｡蜈・焚縺・or3蜃ｺ縺ｪ縺・→縺・
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input_data.dim() == 2:            # input縺ｮ谺｡蜈・焚縺・縺ｮ譎・
            input_data = input_data.unsqueeze(1)  # 蠖｢迥ｶ縺ｮn逡ｪ逶ｮ縺・縺ｫ縺ｪ繧九ｈ縺・↓谺｡蜈・ｒ霑ｽ蜉(莉雁屓縺ｮ蝣ｴ蜷・=1)

        batch_size = input_data.size(0)
        channels = input_data.size(1)
        num_sample = input_data.size(2)
        # print(f'input.size:{input.size()}') # 谺｡蜈・焚縺ｮ遒ｺ隱・[1,1,128000]
        rest = self.win - (self.stride + num_sample % self.win) % self.win
        # print(f'rest:{rest}')

        if rest > 0:
            zero_tensor=torch.zeros(batch_size, channels, rest) # tensor蝙九・3谺｡蜈・・蛻励ｒ菴懈・[batch_size, 1, rest]
            # print(f'zero_tensor.size:{zero_tensor.size()}')
            # pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            pad = Variable(zero_tensor).type(input_data.type())
            # print(f'pad.size():{pad.size()}')
            input_data = torch.cat([input_data, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, self.channel, self.stride)).type(input_data.type())
        # print(f'pad_aux.size():{pad_aux.size()}')
        output = torch.cat([pad_aux, input_data, pad_aux], 2)

        # print('patting\n')
        return output, rest

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        蟄ｦ鄙偵・謇矩・

        Parameters
        ----------
        input_data(tensor):蜈･蜉帙ョ繝ｼ繧ｿ

        Returns
        -------
        decoder_output(tensor):ConvTasNet縺ｮ蜃ｺ蜉帑ｿ｡蜿ｷ(謗ｨ貂ｬ蛟､)
        """
        # print(f'type(input_data):{type(input_data)}')
        # print(f'input_data.shape:{input_data.shape}') #input_data.shape[1,繝√Ε繝ｳ繝阪Ν謨ｰ,髻ｳ螢ｰ髟ｷ]
        """padding"""
        input_patting, rest = self.patting_signal(input_data)
        # print(f'type(input_patting):{type(input_patting)}')
        # print(f'input_patting.shape:{input_patting.shape}')
        batch_size = input_patting.size(0)
        # print(f'batch_size:{batch_size}')
        """encoder"""
        encoder_output = self.encoder(input_patting)  # B, N, L
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')
        """generate masks (separation)"""
        masks = torch.sigmoid(self.TCN(encoder_output)).view(batch_size, self.num_speaker, self.encoder_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        masked_output = encoder_output.unsqueeze(1) * masks  # B, C, N, L
        # print(f'type(masked_output):{type(masked_output)}')
        # print(f'masked_output.shape:{masked_output.shape}')
        """decoder"""
        reshape_masked_output = masked_output.view(batch_size * self.num_speaker, self.encoder_dim, -1)
        decoder_output = self.decoder(reshape_masked_output)  # B*C, 1, L
        # print(f'0:type(decoder_output):{type(decoder_output)}')
        # print(f'0:decoder_output.shape:{decoder_output.shape}')
        decoder_output = decoder_output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
        # print(f'1:type(decoder_output):{type(decoder_output)}')
        # print(f'1:decoder_output.shape:{decoder_output.shape}')
        decoder_output = decoder_output.view(batch_size, self.num_speaker, -1)  # B, C, T
        # print(f'2:type(decoder_output):{type(decoder_output)}')
        # print(f'2:decoder_output.shape:{decoder_output.shape}')
        return decoder_output

class separate_ConvTasNet(nn.Module): # 髻ｳ貅仙・髮｢
    def __init__(self, enc_dim=512, feature_dim=128, sampling_rate=16000, win=2, layer=8, stack=3,
                 kernel=3, num_spk=2, causal=False):    #num_spk=1繧ゅ→繧ゅ→縺ｮ繧・▽
        super(separate_ConvTasNet, self).__init__()

        """hyper parameters"""
        self.num_spk = num_spk                      # 隧ｱ閠・焚 2or3
        self.enc_dim = enc_dim                      # 繧ｨ繝ｳ繧ｳ繝ｼ繝縺ｫ蜈･蜉帙☆繧区ｬ｡蜈・焚
        self.feature_dim = feature_dim              # 迚ｹ蠕ｴ谺｡蜈・焚
        self.win = int(sampling_rate * win / 1000)  # 遯馴聞
        self.stride = self.win // 2                 # 逡ｳ縺ｿ霎ｼ縺ｿ蜃ｦ逅・↓縺翫￠繧九ヵ繧｣繝ｫ繧ｿ縺檎ｧｻ蜍輔☆繧句ｹ・
        self.layer = layer                          # 螻､謨ｰ
        self.stack = stack                          # 繧ｹ繧ｿ繝・け謨ｰ
        self.kernel = kernel                        # 繧ｫ繝ｼ繝阪Ν
        self.causal = causal                        #
        self.channel = 1                            # 繝√Ε繝ｳ繝阪Ν謨ｰ

        """input encoder"""
        self.encoder = nn.Conv1d(in_channels=self.channel,  # 蜈･蜉帙ョ繝ｼ繧ｿ縺ｮ谺｡蜈・焚 #=1繧ゅ→繧ゅ→縺ｮ繧・▽
                                 out_channels=self.enc_dim, # 蜃ｺ蜉帙ョ繝ｼ繧ｿ縺ｮ谺｡蜈・焚
                                 kernel_size=self.win,      # 逡ｳ縺ｿ霎ｼ縺ｿ縺ｮ繧ｵ繧､繧ｺ(豕｢蠖｢鬆伜沺縺ｪ縺ｮ縺ｧ遯馴聞縺ｪ縺ｮ?)
                                 bias=False,                # 繝舌う繧｢繧ｹ縺ｮ譛臥┌(蜃ｺ蜉帙↓蟄ｦ鄙貞庄閭ｽ縺ｪ繝舌う繧｢繧ｹ縺ｮ霑ｽ蜉)
                                 stride=self.stride)        # 逡ｳ縺ｿ霎ｼ縺ｿ蜃ｦ逅・・遘ｻ蜍募ｹ・

        """TCN separator"""
        self.TCN = TCN(input_dim=self.enc_dim,                   # 蜈･蜉帙ョ繝ｼ繧ｿ縺ｮ谺｡蜈・焚
                       output_dim=self.enc_dim * self.num_spk,   # 蜃ｺ蜉帙ョ繝ｼ繧ｿ縺ｮ谺｡蜈・焚
                       BN_dim=self.feature_dim,
                       hidden_dim=self.feature_dim * 4,
                       layer=self.layer,                         # 螻､謨ｰ
                       stack=self.stack,                         # 繧ｹ繧ｿ繝・け謨ｰ
                       kernel=self.kernel,                       # 繧ｫ繝ｼ繝阪Ν繧ｵ繧､繧ｺ
                       causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

        """output decoder"""
        self.decoder = nn.ConvTranspose1d(self.enc_dim, # 蜈･蜉帶ｬ｡蜈・焚
                                          1,    # 蜃ｺ蜉帶ｬ｡蜈・焚 隧ｱ閠・・莠ｺ謨ｰ蛻・・蜉帙☆繧・[1,髻ｳ螢ｰ髟ｷﾃ苓ｩｱ閠・焚]
                                          self.win, # 繧ｫ繝ｼ繝阪Ν繧ｵ繧､繧ｺ
                                          bias=False,   # 繝舌う繧｢繧ｹ縺ｮ譛臥┌(蜃ｺ蜉帙↓蟄ｦ鄙貞庄閭ｽ縺ｪ繝舌う繧｢繧ｹ縺ｮ霑ｽ蜉)
                                          stride=self.stride)   # 逡ｳ縺ｿ霎ｼ縺ｿ蜃ｦ逅・・遘ｻ蜍募ｹ・

    def patting_signal(self, input_data: torch.Tensor) -> tuple:
        """
        蜈･蜉帙ョ繝ｼ繧ｿ繧偵ヱ繝・ぅ繝ｳ繧ｰ竊堤糞縺ｿ霎ｼ縺ｿ蜑阪・谺｡蜈・焚縺ｨ逡ｳ縺ｿ霎ｼ縺ｿ蠕後・谺｡蜈・焚繧貞酔縺倥↓縺吶ｋ縺溘ａ縺ｫ蜈･蜉帙ョ繝ｼ繧ｿ繧・縺ｧ蝗ｲ繧謫堺ｽ・

        Parameters
        ----------
        input_data(tensor[1,繝√Ε繝ｳ繝阪Ν謨ｰ,髻ｳ螢ｰ髟ｷ]):蜈･蜉帙ョ繝ｼ繧ｿ

        Returns
        -------
        output(tensor):蜃ｺ蜉・
        rest:
        """
        # input is the waveforms: (B, T) or (B, 1, T)
        """reshape and padding"""
        if input_data.dim() not in [2, 3]:  # input縺ｮ谺｡蜈・焚縺・or3蜃ｺ縺ｪ縺・→縺・
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input_data.dim() == 2:  # input縺ｮ谺｡蜈・焚縺・縺ｮ譎・
            input_data = input_data.unsqueeze(1)  # 蠖｢迥ｶ縺ｮn逡ｪ逶ｮ縺・縺ｫ縺ｪ繧九ｈ縺・↓谺｡蜈・ｒ霑ｽ蜉(莉雁屓縺ｮ蝣ｴ蜷・=1)

        batch_size = input_data.size(0)
        channels = input_data.size(1)
        num_sample = input_data.size(2)
        # print(f'input_data.size:{input_data.size()}') # 谺｡蜈・焚縺ｮ遒ｺ隱・[1,1,128000]
        rest = self.win - (self.stride + num_sample % self.win) % self.win
        # print(f'rest:{rest}')

        if rest > 0:
            zero_tensor = torch.zeros(batch_size, channels, rest)  # tensor蝙九・3谺｡蜈・・蛻励ｒ菴懈・[batch_size, 1, rest]
            # print(f'zero_tensor.size:{zero_tensor.size()}')
            # pad = Variable(torch.zeros(batch_size, 1, rest)).type(input_data.type())
            pad = Variable(zero_tensor).type(input_data.type())
            # print(f'pad.size():{pad.size()}')
            input_data = torch.cat([input_data, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, self.channel, self.stride)).type(input_data.type())
        # print(f'pad_aux.size():{pad_aux.size()}')
        output = torch.cat([pad_aux, input_data, pad_aux], 2)

        # print('patting\n')
        return output, rest

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        蟄ｦ鄙偵・謇矩・

        Parameters
        ----------
        input_data(tensor):蜈･蜉帙ョ繝ｼ繧ｿ

        Returns
        -------
        decoder_output(tensor):ConvTasNet縺ｮ蜃ｺ蜉・謗ｨ貂ｬ蛟､)
        """
        # print(f'type(input_data):{type(input_data)}')
        # print(f'input_data.shape:{input_data.shape}') #input_data.shape[1,繝√Ε繝ｳ繝阪Ν謨ｰ,髻ｳ螢ｰ髟ｷ]
        """padding"""
        input_patting, rest = self.patting_signal(input_data)
        # print(f'type(input_patting):{type(input_patting)}')
        # print(f'input_patting.shape:{input_patting.shape}')
        batch_size = input_patting.size(0)
        # print(f'batch_size:{batch_size}')

        """encoder"""
        encoder_output = self.encoder(input_patting)  # B, N, L
        # print(f'type(encoder_output):{type(encoder_output)}')
        # print(f'encoder_output.shape:{encoder_output.shape}')

        """generate masks (separation)"""
        masks = torch.sigmoid(self.TCN(encoder_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        # print(f'type(masks):{type(masks)}')
        # print(f'masks.shape:{masks.shape}')
        masked_output = encoder_output.unsqueeze(1) * masks  # B, C, N, L
        # print(f'type(masked_output):{type(masked_output)}')
        # print(f'masked_output.shape:{masked_output.shape}')

        """decoder"""
        decoder_output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        # print(f'0:type(decoder_output):{type(decoder_output)}')
        # print(f'0:decoder_output.shape:{decoder_output.shape}')
        decoder_output = decoder_output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
        # print(f'1:type(decoder_output):{type(decoder_output)}')
        # print(f'1:decoder_output.shape:{decoder_output.shape}')
        decoder_output = decoder_output.view(batch_size, self.num_spk, -1)  # B, C, T
        # print(f'2:type(decoder_output):{type(decoder_output)}')
        # print(f'2:decoder_output.shape:{decoder_output.shape}')

        return decoder_output

