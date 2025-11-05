from eval_asr_from_npz import eval_asr_from_npz
from eval_quality_from_npz import eval_quality_from_npz
from eval_fid_from_npz import eval_fid_from_npz
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def eval_all(file_name=None, re_logger=True, quant=False, eval_model=None):
    eval_quality_from_npz(file_name=file_name, re_logger=re_logger, quant=quant)
    eval_asr_from_npz(file_name=file_name, re_logger=re_logger, quant=quant, eval_model=eval_model)
    eval_fid_from_npz(file_name=file_name, re_logger=re_logger, quant=quant)


if __name__ == "__main__":
    dir_list = [

    ]

    for dir in dir_list:
        print("*********************** Eval All {} ***********************".format(dir))
        eval_all(file_name=dir, re_logger=True, quant=False, eval_model="inception_v3")
        print("************************ Done ************************")

