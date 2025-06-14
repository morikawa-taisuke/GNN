# coding:utf-8
import torch
from tqdm import tqdm
from mymodule import my_func
import soundfile as sf

def sisdr_evaluation(target_data, estimation_data, eps=1e-8):
    """SI-SDRを算出
    
    :param target_path: 正解データのパス
    :param estimation_path: モデル適用後データのパス
    :return sisdr_score: sisdr値
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if target_data.shape != estimation_data.shape:
        raise RuntimeError(
            f"Dimention mismatch when calculate si-sdr, {target_data.shape} vs {estimation_data.shape}"
        )

    target__zm = torch.from_numpy(target_data.astype('float32')) - torch.mean(torch.from_numpy(target_data.astype('float32')), dim=-1, keepdim=True)
    estimation__zm = torch.from_numpy(estimation_data.astype('float32')) - torch.mean(torch.from_numpy(estimation_data.astype('float32')), dim=-1, keepdim=True)
    t = torch.sum(target__zm * estimation__zm, dim=-1,keepdim=True) * estimation__zm / torch.sum(estimation__zm * estimation__zm, dim=-1,keepdim=True)
    sisdr_score = 20 * torch.log10(eps + l2norm(t) / (l2norm(t - target__zm) + eps))
    return sisdr_score

def sisdr_main(target_dir, estimation_dir, out_path):
    """ sisdrを計算する
    
    :param target_path: 正解データのパス
    :param estimation_path: モデル適用後データのパス
    :return sisdr_score: sisdr値
    """

    """ file list 取得 """
    target_list = my_func.get_file_list(target_dir)
    estimation_list = my_func.get_file_list(estimation_dir)

    #my_func.remove_file(file_name)
    """ 出力用のディレクトリーがない場合は作成 """
    my_func.make_dir(out_path)
    """ ファイルの書き込み """
    with open(out_path, 'w') as out_file:
        out_file.write('target_name,estimation_name,SI-SDR_score\n')
    sisdr_sum = 0

    for target_file, estimation_file in tqdm(zip(target_list, estimation_list)):
        """ ファイル名の取得 """
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)
        """ データの読み込み """
        target_data, fs = sf.read(target_file)
        estimation_data, fs = sf.read(estimation_file)
        # print(f'target_data.shape:{target_data.shape}')
        # print(f'estimation_data.shape:{estimation_data.shape}')
        # print(f'target_data:{type(target_data)}')
        # print(f'estimation_data:{type(estimation_data)}')

        if len(target_data) > len(estimation_data):
            target_data = target_data[:len(estimation_data)]
        # elif len(target_data) < len(estimation_data):
        else:
            estimation_data = estimation_data[:len(target_data)]

        """ 型の調整 """
        # target_data = torch.from_numpy(target_data)
        # estimation_data = torch.from_numpy(np.array(estimation_data, dtype=np.float64))
        # print(f'target_data:{type(target_data)}')
        # print(f'estimation_data:{type(estimation_data)}')
        # print(f'target_data:{target_data.shape}')
        # print(f'estimation_data:{estimation_data.shape}')
        """ sisdrの計算 """
        sisdr_score = sisdr_evaluation(target_data=target_data, estimation_data=estimation_data)
        # print(f'sisdr_score:{sisdr_score}')
        # print(f'sisdr_score.dtype:{sisdr_score.dtype}')
        sisdr_score = sisdr_score.detach().numpy()
        sisdr_sum+=sisdr_score
        # print(f'sisdr_score:{sisdr_score}')
        # print(f'sisdr_score.dtype:{sisdr_score.dtype}')
        """ スコアの書き込み """
        with open(out_path, 'a') as out_file:                              # ファイルオープン
            text = f'{target_name},{estimation_name},{sisdr_score}\n'
            out_file.write(text)                                        # 書き込み
    """average"""
    sisdr_ave = sisdr_sum/len(estimation_list)
    with open(out_path, 'a') as out_file:                              # ファイルオープン
        text = f'average,,{sisdr_ave}\n'
        out_file.write(text)                                        # 書き込み
if __name__ == '__main__':
    # target_dir = '../../../sound_data/ss/mix_data/clean'
    # estimation_dir = '../../../sound_data/ss/result/low3_hi4/subband/hoth_10dB'
    #
    # sisdr_main(target_dir = target_dir,
    #            estimation_dir = estimation_dir,
    #            out_path='../../../sound_data/ss/result/low3_hi4/subband/hoth_10dB/_sisdr.csv')

    # loss_list = ['SISDR', 'waveMSE', 'stftMSE']
    noise_list = ['white', 'hoth']
    learning_list = ['noise_only_delay', 'reverbe_only_delay', 'noise_reverbe_delay']
    # for loss in loss_list:
    for noise in noise_list:
        # for is_wave in wave_type_list:
            # sisdr_main(target_dir=f'../../../sound_data/mix_data/02_08/JA_{noise}_10dB_07sec/test/clean',
            #            estimation_dir=f'../../../sound_data/mix_data/02_08/JA_{noise}_10dB_07sec/test/{is_wave}',
            #            out_path=f'./02_08/JA_{noise}_10dB_07sec_{is_wave}_noise_data.csv')
        sisdr_main(target_dir=f'../../../sound_data/mix_data/02_08/JA_{noise}_10dB_07sec/test/clean',
                   estimation_dir=f'../../../sound_data/mix_data/02_08/JA_{noise}_10dB_07sec/test/clean',
                   out_path=f'./02_08/JA_{noise}_10dB_07sec_clean_data.csv')



"""
    calc_sdr('../../data_sample/test/clean_0.5_test',
             '../../data_sample/test/tasnet_mix_re_03-k_2030-re_0-pre01',
             'tasnet_mix_re_03-k_2030-re_0-pre01')

    calc_sdr('../../data_sample/test/clean_0.5_test',
             '../../data_sample/test/tasnet_mix_re_03-k_2030-re_0-pre05',
             'tasnet_mix_re_03-k_2030-re_0-pre05')

    calc_sdr('../../data_sample/test/clean_0.5_test',
             '../../data_sample/test/tasnet_mix_re_03-k_2030-re_0-pre097',
             'tasnet_mix_re_03-k_2030-re_0-pre097')
"""