from mymodule import const
import os


speech_type = "test"  # train test val
speech_dir = os.path.join(const.SAMPLE_DATA_DIR, "speech", "DEMAND", speech_type)
noise_dir = os.path.join(const.SAMPLE_DATA_DIR, "noise", "hoth.wav")
ir_dir = os.path.join(const.MIX_DATA_DIR, "reverb_encoder", "reverb_features", "signal")
output_audio_dir = os.path.join(const.MIX_DATA_DIR, "reverb_encoder", "mix_wav", speech_type)
output_csv = os.path.join(const.MIX_DATA_DIR, "reverb_encoder", "mix_wav", f"{speech_type}.csv")
snr=5